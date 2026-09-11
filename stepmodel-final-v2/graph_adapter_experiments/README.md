# Graph Prefix Adapter experiments

See **`README_STANDALONE.md`** — that's the active version of this
experiment: a fully self-contained GNN + adapter, trained from scratch,
with zero code dependency on the main `stepmodel-final` pipeline (`core/`,
`data_prep/`, `training/`, `eval/`) and zero dependency on any of its
trained checkpoints. It reads only the raw `"graph"` field out of your
existing `input/{train,test}.json` — never the strategy text, never the
main pipeline's classes.

`legacy_core_coupled/` holds the previous version of this folder, which
deliberately reused the main pipeline's `Stage1Classifier`,
`GraphPrefixAdapter`, and Stage-2 LoRA checkpoint (graph embedding fused
with strategy-text embedding before the LLM ever saw it). Kept for
reference only — nothing in the active experiment imports from it.
