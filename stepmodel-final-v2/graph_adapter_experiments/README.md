# Graph Prefix Adapter Reliability Suite

Tests whether the Graph Prefix Adapter in your pipeline actually gives the
LLM usable graph-structure information, using controlled interventions
(wrong graph, no graph, shuffled nodes, etc.) rather than a single accuracy
number. See `RESEARCH_PLAN.md` for the full reasoning and how to read the
results — read that first, this file is just the "how to run it" reference.

## Install

Copy this whole folder (`graph_adapter_experiments/`) into the ROOT of your
`stepmodel-final` repo, next to `run.py` (works with either the original
flat layout or the restructured `core/ data_prep/ training/ eval/` layout —
every script here auto-detects which one it's sitting in).

```
stepmodel-final/
├── run.py
├── core/  data_prep/  training/  eval/        (your existing pipeline)
├── checkpoints/  input/  data/  ...
└── graph_adapter_experiments/                  <- this folder
    ├── common.py
    ├── build_structure_tasks.py
    ├── train_structure_adapter.py
    ├── train_multitask_adapter.py
    ├── run_reliability_suite.py
    ├── analyze_results.py
    ├── evaluate_structure_impact_on_step_task.py
    ├── RESEARCH_PLAN.md
    └── README.md   (this file)
```

No new dependencies beyond what your pipeline already needs
(`torch`, `torch_geometric`, `transformers`, `peft`). `matplotlib` is
optional — `analyze_results.py` will skip charts and still write the
Markdown report if it's not installed.

Everything below assumes you already have `checkpoints/stage1_gnn_classifier.pt`
and `checkpoints/stage2_qwen_lora/` (i.e. Stage 1 and Stage 2 have already
been trained) — this suite builds on top of those, it doesn't retrain them.

## Run order

### 1. Build the structure-probe task datasets (fast, no GPU needed)

```bash
python graph_adapter_experiments/build_structure_tasks.py
```

Writes `graph_adapter_experiments/structure_tasks/{train,held_out}.jsonl`
and `split_manifest.json`. Check the printed machine counts and the
"LEAKAGE" assertion — it will hard-fail if any machine ends up in both
splits, which should never happen but is worth having a tripwire for.

### 2. Run the reliability suite on your EXISTING Stage-2 checkpoint first

This is the most informative first run: it tells you what your current,
already-trained model does before you train anything new.

```bash
python graph_adapter_experiments/run_reliability_suite.py \
    --checkpoint checkpoints/stage2_qwen_lora \
    --split held_out --max_items 150
```

Writes `graph_adapter_experiments/results/raw_results_stage2_qwen_lora.jsonl`.
Takes roughly as long as one `eval/evaluate.py --model llm` run per
condition (8 conditions × up to 150 items = up to 1200 generations) — use
`--max_items` and `--conditions real,zero,wrong_graph` to shrink this for a
first pass.

### 3. Analyze it

```bash
python graph_adapter_experiments/analyze_results.py \
    --results graph_adapter_experiments/results/raw_results_stage2_qwen_lora.jsonl
```

Writes `results/REPORT.md` (+ `results/chart_*.png` if matplotlib is
installed) with per-task tables, paired significance tests, and a verdict
per task. **This alone already answers most of your original question**
using your existing checkpoint — no new training required.

### 4. Train a structure-focused adapter (optional, if step 3 says "no
   evidence of graph understanding" and you want to see if it's trainable
   at all)

```bash
python graph_adapter_experiments/train_structure_adapter.py \
    --init_from checkpoints/stage2_qwen_lora \
    --out_dir   checkpoints/graph_structure \
    --steps 1500
```

Prints held-out structure score every `--eval_every` steps and only saves
`checkpoints/graph_structure/best/` when a checkpoint actually beats the
Stage-2 starting point — if it never does, the script says so explicitly
rather than silently shipping the last step (same fix already applied to
`training/stage3_grpo_rl.py`).

Then re-run steps 2–3 pointed at this new checkpoint:

```bash
python graph_adapter_experiments/run_reliability_suite.py \
    --checkpoint checkpoints/graph_structure/best --split held_out
python graph_adapter_experiments/analyze_results.py \
    --results graph_adapter_experiments/results/raw_results_best.jsonl \
              graph_adapter_experiments/results/raw_results_stage2_qwen_lora.jsonl
```

(Passing both files to `analyze_results.py` puts both checkpoints in one
report so you can compare before/after directly.)

### 5. Does structure training affect the real step-prediction task?

```bash
python graph_adapter_experiments/evaluate_structure_impact_on_step_task.py
```

Runs your actual `eval/evaluate.py`'s `eval_llm()` (unmodified) on the
Stage-2 baseline and `checkpoints/graph_structure/best`, and prints a
side-by-side delta table.

### 6. Multi-task training (structure + step-prediction together)

```bash
python graph_adapter_experiments/train_multitask_adapter.py \
    --init_from checkpoints/stage2_qwen_lora \
    --out_dir   checkpoints/multitask \
    --steps 2000 --structure_frac 0.3
```

Then repeat steps 2–3 and step 5 pointed at `checkpoints/multitask/best`
to see both sides (structure understanding AND step-prediction accuracy)
for the jointly-trained model.

## Tuning knobs worth sweeping

- `--structure_frac` in `train_multitask_adapter.py` (try 0.1, 0.3, 0.5) —
  how much structure training is "too much" before it starts costing
  step-prediction accuracy?
- `--tasks` in `build_structure_tasks.py` / `run_reliability_suite.py` —
  run `graph_aggregate` alone to isolate the "fair" pooled-embedding test
  from the harder per-node tasks (see RESEARCH_PLAN.md §2).
- `--conditions` in `run_reliability_suite.py` — for a quick sanity check,
  `real,zero,wrong_graph` alone (3 conditions instead of 8) is enough to
  get the headline verdict much faster.

## Interpreting a null result

If every task comes back "❌ No evidence of graph understanding" even after
structure training, that is itself a real, useful research finding — it
would mean the pooled single-vector → 8-token bottleneck (see
RESEARCH_PLAN.md §2) is the limiting factor, not the training procedure.
The architectural fix in that case would be a per-node or per-subgraph
prefix representation (e.g. a small fixed number of top-attended node
embeddings alongside the pooled summary) rather than more training on the
current adapter.
