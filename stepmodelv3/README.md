# stepmodelv3 — GRPO RL pipeline (corrected)

## What I found

Your `stage1_grpo_rl.py` / `evaluate.py` assumed a data schema that **doesn't match**
your actual `train.json` / `test.json`:

| Assumed key | Real key | Real content |
|---|---|---|
| `Machine` | `machine` | ok, just wrong case |
| `Graph` | `graph` | ok, just wrong case |
| `step_label` (from a fixed 10-item list) | `gold_new_step` | free text, 46 raw variants |
| `mcp_labels` (list) | `gold_mcp_tasks` | a string like `"Tool: desc; Tool2: desc2"` (not JSON) |

With the original code, `gold_mcp_tasks` failed `json.loads`/dict-parsing on **1413/1728**
rows, and step matching against the hardcoded `STEP_LABELS` (`"google search"`,
`"enumerate further"`, ...) would almost never exact-match your actual `gold_new_step`
strings (e.g. `"Enumerate further on the HTTP service to find software versions..."`).
Training reward would have been ~0 almost the whole time.

## What I changed

- **`data_utils.py`** — real loader for `machine`/`graph`/`gold_new_step`/
  `gold_step_explanation`/`gold_mcp_tasks`. I clustered your data and derived:
  - **10 canonical step categories** (`recon_scan`, `enumerate_further`,
    `enumerate_website`, `enumerate_domain`, `explore_files`,
    `source_code_review`, `google_search`, `exploit`, `analyze_outcomes`,
    `end_task`) — covers 1721/1728 train rows (99.6%).
  - **18 canonical MCP tools** (`nmap`, `metasploit`, `dirbuster`,
    `john_the_ripper`, `smb_client`, `sqlmap`, `hydra`, `hashcat`, `netexec`,
    `git_dumper`, `burp_suite`, `ftp_client`, `responder`, `autopsy`, `netcat`,
    `google_search`, `web_page_interaction`, `interactive_cli`) — covers
    >99% of tool mentions.
  - A robust `parse_mcp_tasks()` that handles both the semicolon format and
    the rarer python-dict-repr format.
  - `compute_reward()`: format (0.10) + step (0.35: 0.6×category match +
    0.4×text similarity, so it rewards both the right *kind* of step and
    reproducing graph-specific detail like the service/IP) + MCP set-F1
    (0.25) + explanation similarity (0.30).
- **`prompts.py`** — the raw `graph` JSON is serialized with `json.dumps()` and
  put straight into the prompt text — **no embedding model**, exactly as you
  asked. It also includes the `new_strategy`/`strategy_explanation` fields as
  reference context (they look like a heuristic's draft guess) since they're
  present in every row and give the model useful signal.
- **`train_grpo.py`** — same GRPO mechanics as your original (LoRA policy +
  frozen LoRA reference, group-relative advantage, PPO-clipped policy
  gradient + KL penalty — that part of your original script was correctly
  implemented), rewired onto the corrected data/reward layer. Also bumped
  LoRA rank (16→32) and target modules (attention-only → +MLP) and sampling
  temperature (0.3→0.6) since GRPO needs real diversity across the group to
  get a non-degenerate advantage signal.
- **`evaluate.py`** — rewired onto the same canonical categories/tools, uses
  greedy decoding (not sampling) for reproducible accuracy numbers, and adds
  a "step + MCP joint accuracy" number.

I ran the full data-loading → reward → metrics pipeline against your actual
`train.json` (1728 rows) and `test.json` (268 rows) in this sandbox (mocking
only the LLM generation calls, since I don't have a GPU or Qwen3-14B here) —
everything parses and runs cleanly end to end.

## How to run

```bash
pip install -r requirements.txt
python train_grpo.py                       # writes checkpoints/stage1_grpo_rl
python evaluate.py --save-explanations out.csv
```

This needs a GPU with enough VRAM for Qwen3-14B in bf16 (policy + frozen
reference model both loaded — realistically 2×24GB+ or one 80GB card) and
internet access to `huggingface.co` to pull the model and
`sentence-transformers/all-MiniLM-L6-v2`. Neither is available in this
sandbox, so I could not execute an actual training run for you here — only
validate that every non-GPU code path is correct against your real data.

## About the 85–90% target

I can't make a script *guarantee* an accuracy number — that depends on real
training against real GPUs, not just hyperparameters. What I can tell you
honestly:

- **Step category accuracy (10 classes)** is the most learnable of the three:
  your data is highly repetitive (46 raw strings → 10 clusters, some clusters
  with 600+ examples), so 85-90%+ is a realistic target with enough GRPO
  steps.
- **MCP tool set-F1** is harder — 18 possible tools per row, often
  multi-label, several tools have very few examples (`autopsy`, `responder`,
  `burp_suite` have ≤1 each in train). Rare tools will likely underperform no
  matter how long you train; that's a data-imbalance ceiling, not a bug.
- **Explanation quality** doesn't have a natural "% correct" — I report BLEU/
  ROUGE-L/BERTScore-F1 instead of forcing it into an accuracy number, since a
  paraphrased-but-correct explanation shouldn't be scored as wrong.

Recommendation: run `train_grpo.py`, watch the `step_cat_acc` / `mcp_f1` /
`exp_sim` columns it prints every 10 steps, and extend `NUM_STEPS` (or widen
`LORA_R`/`GROUP_SIZE`) if they're still climbing rather than assuming a fixed
step count will land in a specific range.
