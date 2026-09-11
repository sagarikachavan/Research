# Graph Prefix Adapter — standalone experiment

**Goal:** find out whether an LLM can actually read graph *structure* out
of soft-prompt tokens produced by a GNN — and, specifically, whether it
notices when it's given the *wrong* graph for a query it's otherwise seen
before. This is deliberately isolated from the main `stepmodel-final`
pipeline: nothing here is trained on, or dependent on, the main pipeline's
strategy-fused GNN, its Stage-2 LoRA checkpoint, or any of its code.

## What's isolated vs what's shared

| | Main pipeline | This experiment |
|---|---|---|
| GNN | `core/graph_encoder.py` `Stage1Classifier` — graph **+ text** ("New strategy"/"Strategy explanation") fused via cross-attention before the LLM ever sees it | `structure_gnn.py` `StructureGNN` — graph **only**, trained from scratch here, no text input at all |
| Node features | 384-dim text embedding of node title + type one-hot + degree | Type one-hot + status one-hot + degree/BFS-depth/is-start — **no text/title embedding anywhere** |
| Adapter | `training/stage2_sft_qwen.py` `GraphPrefixAdapter`, loaded from `checkpoints/stage2_qwen_lora/graph_adapter.pt` | `prefix_adapter.py` `GraphPrefixAdapter` — same idea, separate class, always trained fresh |
| LLM checkpoint | `checkpoints/stage2_qwen_lora` / `stage3_qwen_grpo` (trained on your full pipeline) | A fresh base model + optional fresh LoRA (`--use_lora`), never any main-pipeline checkpoint |
| Code imports | — | **Zero** imports from `core/`, `data_prep/`, `training/`, or `eval/` |
| Data | `input/train.json` / `input/test.json` | Same files, but **only the `"graph"` field is read** — `new_strategy`, `strategy_explanation`, `gold_*` are never touched |

The previous version of this folder (`common.py`, `train_structure_adapter.py`,
`train_multitask_adapter.py`, `run_reliability_suite.py`,
`evaluate_structure_impact_on_step_task.py`, `analyze_results.py`, plus its
old `checkpoints/`, `structure_tasks/`, `results/`) explicitly reused the
main pipeline's `Stage1Classifier`, `GraphPrefixAdapter`, and Stage-2 LoRA
checkpoint — the opposite of what's needed here. They've been moved,
unmodified, into `legacy_core_coupled/` for reference; nothing in this
README or the scripts below touches them.

## Files

- `standalone_config.py` — every path/hyperparameter, no external imports.
- `graph_json.py` — raw graph JSON → structural-only features. No text.
- `structure_gnn.py` — `StructureGNN`, a small GATv2 encoder, random init.
- `prefix_adapter.py` — `GraphPrefixAdapter`, graph embedding → K soft tokens.
- `build_probe_tasks.py` — builds the QA tasks below from the raw graphs.
- `probe_prompts.py` — shared prompt/target/scoring logic.
- `train_adapter.py` — trains `StructureGNN` + `GraphPrefixAdapter` (+ optional fresh LoRA) end-to-end through a base LLM's language-modeling loss.
- `eval_right_vs_wrong_graph.py` — the actual right-graph-vs-wrong-graph test.

## The tasks

Every task is built purely from `graph["nodes"]`/`graph["edges"]`. Node ids
are anonymized to `N0, N1, N2, ...` per graph so nothing can be answered by
pattern-matching a revealing id string or a title.

- `adjacency`, `node_type`, `edge_type`, `two_hop`, `graph_aggregate` —
  standard structural-QA probes (same spirit as before, just self-contained
  and title-free now).
- **`graph_consistency`** — this is the one that directly matches what you
  described: a claim like *"Node N2 is directly connected to node N5."*
  with a gold `true`/`false` label for its own graph. At train and eval
  time, the identical claim is also served with a **different graph's**
  soft-prompt tokens swapped in — in which case the correct answer is
  always `false`, regardless of the claim's original label, because the
  claim no longer describes the graph actually being shown.

  - Same query + right graph → answer the claim correctly.
  - Same query + wrong graph → answer `false` (recognize the mismatch).

  `train_adapter.py --real_frac 0.5` trains on a 50/50 mix of real-graph
  and decoy-graph `graph_consistency` items by default (tune with
  `--real_frac`). `eval_right_vs_wrong_graph.py` reports both numbers
  separately so you can see the real/wrong-graph trade-off directly,
  plus, for the other tasks, how often the answer *changes* at all when
  the graph is swapped (a model reading the tokens has no reason to
  repeat the same adjacency/type answer for an unrelated graph).

## Run order

**One script runs all three steps: `run_all.py`.**

```bash
# Point this at your existing input JSON (only the "graph" field is read)
export STANDALONE_INPUT_TRAIN_JSON=/path/to/stepmodel-final/input/train.json
export STANDALONE_INPUT_TEST_JSON=/path/to/stepmodel-final/input/test.json

python run_all.py
```

That runs, in order: `build_probe_tasks.py` → `train_adapter.py` →
`eval_right_vs_wrong_graph.py`, and prints the reliability report at the
end. Useful flags (all forward to the underlying script):

```bash
python run_all.py --steps 3000 --use_lora            # longer run, fresh LoRA too
python run_all.py --model_name Qwen/Qwen2.5-3B-Instruct
python run_all.py --skip_build                        # reuse existing standalone_tasks/*.jsonl
python run_all.py --skip_train --checkpoint standalone_checkpoints/run1/best   # eval only
```

Or run each step by hand (e.g. to inspect intermediate output, or re-run
just one step) — this is exactly what `run_all.py` chains together:

```bash
# 1. Build the anonymized structural + consistency tasks (fast, no GPU)
python build_probe_tasks.py
# -> standalone_tasks/{train,held_out}.jsonl, split_manifest.json
#    (machine-level split; the script asserts no leakage)

# 2. Train the GNN + adapter (fresh LLM by default: Qwen/Qwen2.5-1.5B-Instruct;
#    override with --model_name). LLM stays frozen unless you pass --use_lora.
python train_adapter.py --steps 1500 --eval_every 150
# -> standalone_checkpoints/run1/best/{structure_gnn.pt, graph_adapter.pt, meta.json}

# 3. The actual right-graph vs wrong-graph test
python eval_right_vs_wrong_graph.py --checkpoint standalone_checkpoints/run1/best
```

## Interpreting the result

- High `graph_consistency` "right graph → true" **and** "wrong graph →
  false" (both well above chance, and comparable to each other) is the
  signature of the adapter genuinely carrying usable structure: the LLM
  is conditioning its answer on the tokens, not on the claim text or a
  fixed default.
- High "right graph" but the "wrong graph" number also stays high (i.e.
  it says `true` regardless of which graph it's shown) means the LLM is
  ignoring the graph tokens and just pattern-matching the claim text —
  the tokens aren't doing anything.
- If `train_adapter.py` reports "held-out score never improved" even
  after training, that's itself a real finding: it says the pooled
  single-vector → K-token bottleneck may be too narrow for the LLM to
  decode fine-grained structure from, independent of the main pipeline's
  training procedure — the architectural fix would be a per-node or
  per-subgraph prefix (a handful of top-attended node embeddings
  alongside the pooled summary) rather than more training on this
  single-vector design.

## Requirements

Same as the rest of the pipeline: `torch`, `torch_geometric`,
`transformers`, `peft` (only if `--use_lora`). No `matplotlib` or anything
else beyond that. Nothing here needs `sentence-transformers` / any text
encoder, since node titles are never embedded.
