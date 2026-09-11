# Graph Prefix Adapter Reliability Suite — Research Plan

## 1. The question

Does the LLM actually understand graph structure when a graph is converted
into 8 soft-prompt tokens by the Graph Prefix Adapter — or does it just use
those tokens as a vague conditioning signal (or ignore them and rely on
other cues in the prompt)?

Your earlier experiment (`GRAPH_PREFIX_ADAPTER_TEST_REPORT.pdf`) got a
single positive data point: 90% recall on adjacency prediction, after
fixing a real bug (random dummy embeddings instead of real graph
embeddings). That result is a good existence proof that *something* can be
learned, but by itself it can't distinguish between three very different
explanations for "90% recall":

1. **Genuine structural decoding** — the adapter's 8 tokens really encode
   this graph's topology, and the LLM learned to read it out.
2. **Memorization** — if the test graph was one of the training graphs (the
   report doesn't say either way), 90% recall could just mean the model
   memorized *this specific graph's* embedding → answer mapping, with zero
   ability to generalize to a graph it hasn't seen.
3. **A shortcut elsewhere in the input** — the query used real node ID
   strings as text (`agent:active:START`), and many machines in this
   dataset share similar graph *templates*. The LLM could be pattern-matching
   on ID-naming conventions it saw repeated across training graphs, with the
   soft-prompt tokens contributing nothing.

This suite is built to tell these apart, and to extend the test to the
tasks your report's "Next steps" section named but didn't run.

## 2. An architectural fact that changes the test design

Reading `graph_encoder.py` directly (not assumed): the vector that actually
becomes the 8 soft-prompt tokens is **not** the GNN's graph embedding on its
own. `Stage1Classifier.encode_and_predict()` fuses the graph embedding `g`
with a **text** embedding `c` of the `New strategy` / `Strategy explanation`
columns via cross-attention + learned gating, and only that fused vector `h`
goes into `GraphPrefixAdapter`.

Practical consequence: if you only ever swap the graph and leave the context
text alone, `real` and `wrong_graph` could score similarly for a reason that
has *nothing* to do with the model reading graph structure — the unchanged
text component `c` could be carrying enough signal on its own. Every
experiment here therefore varies **graph and context independently**
(a 2×2-style design: real/real, wrong-graph/real, real/wrong-context,
wrong-graph/wrong-context), so a claim like "the model understands the
graph" can be pinned specifically on the graph component, not the fusion as
a whole.

The other architectural fact worth knowing: `GraphEncoder.forward()` pools
the entire graph down to **one vector** (mean + max + attention + per-layer
+ Set2Set pooling, concatenated and projected). There is no per-node output
anywhere in the pipeline. This means tasks that ask about a *specific named
node* (adjacency, node type, edge type, two-hop) are asking the model to
recover per-node information from what is architecturally a whole-graph
summary — a demanding, somewhat unnatural ask for this design. The
`graph_aggregate` task (node/edge counts, density, dominant type) is
included specifically because it's the one task family a pooled embedding
could plausibly support without needing per-node addressing at all — it's
the fairer test of "does the pooled vector carry global structural
information," separate from "can the LLM point to a specific node."

## 3. The 8 prefix conditions

| Condition | What changes | What it tests |
|---|---|---|
| `real` | nothing (baseline) | normal operation |
| `wrong_graph` | swap in a different real graph, same context | does the model track *this* graph's structure, or just respond to the question shape? |
| `wrong_context` | swap in different context text, same graph | is any apparent understanding coming from the text half of the fusion instead of the graph half? |
| `wrong_both` | swap both | worst-case control — if this scores as well as `real`, the model may not be conditioning on the input at all |
| `shuffled_nodes` | same graph, but node feature ROWS permuted (structure/edges untouched in the tensor, but the text legend's node-to-position mapping is now scrambled relative to what the GNN encoded) | sensitivity to *which* node is which, vs. just "a graph-shaped input is present" |
| `zero` | prefix = 0 vector | absolute floor — no information at all |
| `noise` | prefix = matched-scale Gaussian noise | is a *present, well-scaled* signal enough, even if it's garbage? (distinguishes "needs SOME embedding" from "needs a MEANINGFUL embedding") |
| `mean_prototype` | prefix = average embedding over many graphs | a generic "graph-shaped prior" — if this does as well as `real`, the model may just be recognizing "yes this is a pentest graph" rather than reading specifics |

## 4. How to read a result

For each task, `analyze_results.py` runs a **paired** permutation test
(same items, real vs. each control) and applies:

- **✅ Evidence of graph-specific understanding**: `real` beats both `zero`
  and `wrong_graph`, significantly (p<0.05) and by a meaningful margin
  (>0.10 absolute score). This is the only pattern that rules out both
  "no signal used at all" and "any real-looking graph would do."
- **⚠ Possible shortcut, not graph-specific**: `real` beats `zero` but not
  `wrong_graph`. The model is using *something* in the fused vector, but
  not this graph's actual structure — check `wrong_context` too; if `real`
  also fails to beat `wrong_context`, the something is very likely the text
  half of the fusion.
- **❌ No evidence of graph understanding**: `real` doesn't clear `zero`
  either. For a per-node task, this may just mean the pooled-embedding
  architecture genuinely can't support this — check whether
  `graph_aggregate` (the fairer test) does any better before concluding
  the adapter is useless.

Held-out numbers are what matter. Every task-building step
(`build_structure_tasks.py`) enforces a machine-level split identical in
spirit to the one already used for Stage 1/2/3, specifically so a high
`real` score can't just be "this exact graph was in the training set."

## 5. The four studies, and which script runs each

1. **Necessity/sufficiency test** (`run_reliability_suite.py` +
   `analyze_results.py`) — the causal intervention battery above, run on
   whichever checkpoint you point it at.
2. **Broader structure tasks** (`build_structure_tasks.py`,
   task families in §2) — node type, edge type, 2-hop reachability, and
   graph-level aggregates, in addition to adjacency.
3. **Impact on the real task** (`evaluate_structure_impact_on_step_task.py`)
   — does structure training help or hurt step-prediction accuracy,
   measured with your existing, unmodified `eval/evaluate.py`.
4. **Multi-task training** (`train_multitask_adapter.py`) — one LoRA fine-tune
   on both objectives together, with checkpoint selection that refuses to
   promote a checkpoint that regresses step-accuracy below the Stage-2
   baseline, so "multitask helped" can't secretly mean "we let the real
   task get worse."

## 6. What this suite deliberately does NOT do

- It does not touch or re-run Stage 1/2/3 training — it only trains
  *additional* adapters that start from your existing Stage-2 checkpoint.
- It does not require GPU-side changes to your existing pipeline code —
  everything here imports your real classes (`Stage1Classifier`,
  `GraphPrefixAdapter`, `eval_llm`) rather than reimplementing them, so
  results are directly attributable to your actual architecture.
- It does not claim a single number "proves" or "disproves" understanding.
  The whole point of the paired-condition design is that the *gap* between
  conditions is the evidence, not any one condition's raw score.
