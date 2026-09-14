# Pipeline architecture

Single source of truth for what this pipeline actually is: the diagram, the
exact dimension/frozen-trainable contract at every stage, the papers behind
each design choice, the reward formula, the GRPO methodology, and the
explanation-evaluation methodology. Replaces the four overlapping
architecture docs that used to live at the repo root (`ARCHITECTURE_DIAGRAM.md`,
`COMPREHENSIVE_DOCUMENTATION.md`, `DOCUMENTATION.md`,
`MULTI_STAGE_ARCHITECTURE.md`) — those had drifted out of sync with each
other and with the code; this one is verified against the current code, file
and line by file and line, as of this write-up.

For the audit trail of *why* the code looks the way it does (bugs found and
fixed, regressions diagnosed, ablations tried) see `STAGE1_IMPROVEMENTS.md`
and `STAGE2_STAGE3_IMPROVEMENTS.md`. This doc is the *current-state* summary;
those are the *history*.

## What's from the paper vs. what's this project's own extension

This pipeline targets the same task as the "Pen-Strategist" paper (Ginige,
Marasinghe, Jain, Seneviratne, "Pen-Strategist: A Reasoning Framework for
Penetration Testing Strategy Formation and Analysis," arXiv:2605.04499) and
is benchmarked directly against its **Step Model** — but the paper's Step
Model is meaningfully simpler than this pipeline, and it's worth being exact
about the difference rather than overclaiming:

| | Pen-Strategist's Step Model | This pipeline |
|---|---|---|
| Input | Strategy + explanation text only | Strategy + explanation text **and** the PTT graph |
| Text encoder | Frozen GPT-2 token embeddings | Same idea, frozen GPT-2 token embeddings |
| CNN | **Two separate** multi-kernel CNN encoders, one per head (§4.2.2) | **One shared** multi-kernel CNN encoder feeding both heads |
| Graph representation | **None at all** — no GNN anywhere in the paper | Typed GINE graph encoder, fused with the semantic representation via cross-attention |
| Output | Step class + MCP tool set (classification only) | Step + MCP + free-text explanation, both as a Stage-1 classifier and as Stage 2/3's generated JSON |
| Stage 2/3 (LLM) | Not part of the Step Model — a *separate* Strategy model (Qwen3-14B + LoRA + GRPO) handles strategy generation, unconditioned on any per-step graph | Qwen3-14B + LoRA, conditioned on the Stage-1 graph embedding via a learned soft-prompt adapter, trained SFT-then-GRPO specifically for step/MCP/explanation prediction |

**The graph encoder, the graph/semantic fusion, the graph-conditioned Qwen
prefix, the whole 3-stage SFT-then-GRPO structure for step/MCP/explanation
prediction, the detailed MCP tool-error breakdown, and the DAPO-style GRPO
refinements are this project's own contributions — none of them are in the
paper.** What *is* from the paper: the frozen-GPT-2-token-embeddings +
multi-kernel-CNN idea for the semantic branch, and the **target numbers**
(Table 3) this pipeline is trying to beat.

## Pipeline diagram

```
┌─────────────────────────┐      ┌──────────────────────────────┐
│        PTT graph        │      │   New strategy + explanation  │
└────────────┬─────────────┘      └───────────────┬────────────────┘
             │                                    │
             ▼                                    ▼
┌──────────────────────────────────── STAGE 1 (trainable end-to-end) ────┐
│  ┌────────────────────┐              ┌─────────────────────────────┐  │
│  │    GINE encoder     │              │   frozen-GPT-2 tokens ->    │  │
│  │  4 layers -> 512-d  │              │   shared multi-kernel CNN   │  │
│  └──────────┬──────────┘              └──────────────┬───────────────┘  │
│             │                                        │                 │
│             │                    ┌───────────────────┘                 │
│             │                    ▼                                     │
│             │        cross-attention fusion -> Step head + MCP head    │
│             │        (Stage 1's own supervised predictions)            │
└─────────────┼───────────────────────────────────────────────────────────┘
              │ 512-d graph embedding ONLY
              ▼
┌──────────────────────────────────── STAGE 2 (Stage 1 frozen) ──────────┐
│  ┌───────────────────────┐         ┌────────────────────────────┐     │
│  │  GraphPrefixAdapter    │ ──────► │      Qwen3-14B + LoRA       │     │
│  │  512-d -> 8 soft tokens│  8 tok  │      supervised fine-tune   │     │
│  │  (trainable)           │         │      (LoRA trainable)       │     │
│  └───────────────────────┘         └────────────────────────────┘     │
└───────────────────────────────────────────┬───────────────────────────┘
                                             │ LoRA + adapter checkpoint
                                             ▼
┌──────────────────────────────────── STAGE 3 (adapter + Stage 1 frozen) ─┐
│  ┌───────────────────────┐         ┌────────────────────────────┐      │
│  │  GraphPrefixAdapter    │ ──────► │      Qwen3-14B + LoRA       │      │
│  │  frozen, from Stage 2  │  8 tok  │      GRPO policy update     │      │
│  └───────────────────────┘         └────────────────────────────┘      │
│                                                                          │
│      reward = 33% step + 33% MCP + 33% explanation + 1% format          │
│      (GDPO-style per-objective z-score normalization within group)      │
└──────────────────────────────────────────┬───────────────────────────────┘
                                            │ final checkpoint
                                            ▼
                              ┌─────────────────────────┐
                              │     Final evaluation      │
                              │  vs. Pen-Strategist paper │
                              └─────────────────────────┘
```

## Frozen / trainable, per stage

| Stage | Module | Status | Where |
|---|---|---|---|
| 1 | GINE graph encoder | trainable | `core/graph_encoder.py::GraphEncoder` |
| 1 | Semantic CNN | trainable | `core/graph_encoder.py::SemanticCNNEncoder` |
| 1 | Cross-attention fusion + Step/MCP heads | trainable | `core/graph_encoder.py::Stage1Classifier` |
| 2 | Stage 1 (all of it) | **frozen** | `training/stage2_sft_qwen.py:685-687`, explicit `requires_grad_(False)` |
| 2 | `GraphPrefixAdapter` | trainable | newly constructed, in `trainable_params` |
| 2 | Qwen3-14B base | frozen | by LoRA construction |
| 2 | Qwen3-14B LoRA | trainable | `get_peft_model`, cast to fp32 |
| 3 | Stage 1 | **frozen** | `training/stage3_grpo_rl.py`, mirrors Stage 2 |
| 3 | `GraphPrefixAdapter` | **frozen** | `TRAIN_ADAPTER=False`; explicit `requires_grad_(False)` **and** excluded from the optimizer's parameter list — belt and suspenders |
| 3 | Qwen3-14B base | frozen | PEFT |
| 3 | Qwen3-14B LoRA | trainable | the **only** thing GRPO updates |

## Dimension / checkpoint flow

| Hop | Shape / value | Source |
|---|---|---|
| PTT node features | 783-d (`TEXT_EMB_DIM=768 + NODE_AUX_DIM=15`) | `core/config.py` |
| GINE hidden width | 384 (`GNN_HIDDEN`), 4 layers (`GNN_LAYERS`) | `core/config.py` |
| GINE output | **512-d** (`GNN_OUT_DIM`) — the sole graph representation exposed to Stage 2/3 | `core/graph_encoder.py`, runtime-asserted at `Stage1Classifier.encode_and_predict` |
| Semantic CNN output | 192-d (`SEMANTIC_CNN_DIM`), kernels `(2,3,4,5,7)` | `core/config.py` |
| Stage-1 fusion width | 768-d (`FUSION_HIDDEN`) | `core/config.py` |
| `GraphPrefixAdapter` input | 512-d (`GRAPH_PREFIX_SRC_DIM = GNN_OUT_DIM`) | `training/stage2_sft_qwen.py:55` |
| `GraphPrefixAdapter` output | `GRAPH_PREFIX_TOKENS=8` tokens at Qwen's hidden size | `core/config.py`, `training/stage2_sft_qwen.py` |
| LoRA rank / alpha / dropout | `r=32, alpha=64, dropout=0.12`, targets all attn+MLP proj matrices | `core/config.py` |
| Stage 1 → 2 checkpoint | `torch.load(STAGE1_CKPT)` → `stage1.load_state_dict(...)` | `training/stage2_sft_qwen.py` |
| Stage 2 → 3 checkpoint | `PeftModel.from_pretrained(base, STAGE2_ADAPTER_DIR)` + `adapter.load_state_dict(.../graph_adapter.pt)` | `training/stage3_grpo_rl.py` |

Both the Stage 1→2 and Stage 2→3 boundaries have a runtime `RuntimeError` if
the dimension doesn't match `GNN_OUT_DIM`/`GRAPH_PREFIX_SRC_DIM` — a shape
drift fails loudly instead of silently misaligning.

## Reward formula (Stage 3)

```
total = 0.01 * format_ok
      + 0.33 * step_r       (exact match, normalized step label)
      + 0.33 * mcp_r        (Jaccard: |pred ∩ gold| / |pred ∪ gold|)
      + 0.33 * exp_r        (0.60*BGE cosine + 0.20*lexical + 0.10*step-support + 0.10*tool-support)
```

`training/stage3_grpo_rl.py::compute_reward`. Equal weight across the three
substantive objectives, tiny format weight (a correctly-formatted-but-wrong
answer must still score poorly) — matches the target design directly.

**Why Jaccard for MCP, not exact match or plain F1**: exact match gives zero
learning signal for a mostly-right tool set (research on multi-label RL
reward design favors partial-credit set metrics); Jaccard and per-row F1 are
closely related and either is defensible — Jaccard was already in place and
is retained.

**Reward normalization**: advantages are **not** a single scalar z-score.
Each of step/MCP/explanation is z-scored independently *within its GRPO
group* (GDPO-style decoupled normalization), then averaged and clamped to
`[-3, 3]` — this prevents one numerically noisy objective (e.g. explanation,
which has more continuous variance than the binary step reward) from
dominating the advantage purely because of its scale, not its actual signal.
This is genuinely research-backed practice for exactly the "don't blindly
average incompatible raw metrics" concern, and was already in place.

**Anti-reward-hacking**: the RL-time explanation reward
(`_deterministic_explanation_score`) is deliberately a *different*,
deterministic/reference-based score from the test-time LLM judge
(`core/llm_judge.py`) — training against the same noisy evaluator you report
results with risks the policy learning the judge's quirks rather than real
explanation quality. Zero-reward-variance groups (all-same-reward, no
learning signal) are rejected and resampled at higher temperature before
being skipped, matching DAPO's "dynamic sampling."

## GRPO methodology actually implemented

- **Correct PPO ratio**: current policy vs. rollout (old) policy — *not*
  vs. the frozen reference, which is a separate KL anchor term
  (`training/stage3_grpo_rl.py`).
- **Dual-clip PPO** (Ye et al., 2020; used in DAPO/verl/TRL GRPO trainers):
  for a negative-advantage sample whose ratio has drifted far above 1,
  standard single-clip PPO doesn't bound the objective from below — floors
  it at `dual_clip_coef * advantage` (`STAGE3_DUAL_CLIP_COEF=3.0`) instead.
  Fixes a real observed incident (`pg_loss` 127 → 8255, val reward
  0.454 → 0.29) — the incident case (`ratio≈22000, adv=-4`) now produces
  `pg_loss=12` instead of `88000` (verified with a synthetic-tensor test).
- **DAPO Clip-Higher** (Yu et al., "DAPO," 2025, arXiv:2503.14476): the
  upper PPO clip bound is widened independently of the lower bound
  (`STAGE3_CLIP_HIGH=0.28` vs. `STAGE3_PPO_CLIP=0.20`) so a completion whose
  probability should increase a lot isn't prematurely capped — DAPO's own
  ablation ties symmetric clipping to entropy collapse. Verified: the lower
  bound and the positive-advantage branch are numerically untouched by this
  change; only ratios in `(1.20, 1.28]` are now allowed through unclipped.
- **Per-micro-batch KL hard cap** (`STAGE3_KL_HARD_CAP=4.0`): raised from an
  earlier, over-tight 1.0 that discarded ~28% of micro-batches (including
  good ones riding along with one noisy one) for no real safety benefit once
  dual-clip already bounds `pg_loss`.
- **Dynamic sampling** (DAPO): zero-reward-variance groups are retried at
  increasing temperature (0.70/0.82/0.95), then skipped if still
  uninformative — already in place.
- **Not adopted**: Dr. GRPO's token-aggregation-bias fix (removing
  response-length normalization from the objective). That fix targets long
  chain-of-thought math generations, where response length varies hugely and
  correlates with correctness; this pipeline's completions are short
  structured JSON (order ~50–260 tokens) from a ~1.5k-row dataset — adopting
  a technique built for a different regime without evidence of the same
  failure mode here would be exactly the kind of blind technique-import this
  project is trying to avoid.

All three GRPO-loss-affecting constants above (`STAGE3_DUAL_CLIP_COEF`,
`STAGE3_CLIP_HIGH`, `STAGE3_KL_HARD_CAP`) are read from `core/config.py` with
a `STAGE3_SAFE_*` environment-variable override, same pattern as every other
Stage-3 hyperparameter — editing `config.py` actually takes effect.

## Explanation-evaluation methodology

Three distinct signals, each labeled by what it actually measures — never
collapsed into one number:

1. **LLM judge** (`core/llm_judge.py`, test-time only) — G-Eval-style
   rubric: relevance / technical_accuracy / completeness / clarity, each
   0–3, with a hard gate (`relevance>=2 AND technical_accuracy>=2 AND
   completeness>=1`) rather than a thresholded average, so compensating
   dimensions can't mask a technically-wrong answer. **Primary** correctness
   signal — the paper itself uses G-Eval for its own explanation-quality
   metric, so this is directly comparable in spirit.
2. **BERTScore** (`eval/evaluate.py::compute_reference_explanation_metrics`,
   test-time, secondary) — reference-based semantic similarity. Added
   because current literature on evaluating generated explanations
   consistently shows semantic-similarity-only metrics can score high
   (F1 0.81–0.90) while missing factual correctness — so it's reported
   alongside, never instead of, the LLM judge. BLEURT deliberately not
   added (stale official checkpoints, not worth the extra dependency for a
   secondary signal); ROUGE/BLEU/METEOR deliberately not added (lexical-
   only metrics are the ones the research specifically flags as failing on
   technical text).
3. **RL-time deterministic score** (`_deterministic_explanation_score`,
   training only) — cheap, reference-aware, deliberately *not* the LLM
   judge (anti-reward-hacking, see above).

## MCP evaluation (multi-label tool prediction)

All computed in `eval/evaluate.py::report_classification`:

- Subset (exact-match) accuracy, exact MCP set match rate
- Micro-F1 (sklearn, pooled over the whole matrix) **and** samples-F1
  (per-row F1 averaged) — **these are different numbers**; samples-F1 is
  what the paper's own "Micro F1" formula (§5.1: "F1 score is computed at
  the per-sample level and then averaged across all samples") actually
  computes, so **samples-F1, not sklearn micro-F1, is the paper-comparable
  number**. Compare against the paper's 0.64 using `mcp_samples_f1`.
- Macro-F1 (per-label average, catches silent failure on rare tools)
- Micro-precision / micro-recall (tells you whether errors skew toward
  extra predicted tools or missing ones)
- Jaccard mean
- **Missing/extra tool counts** (`avg_missing_tools`, `avg_extra_tools`)
  and their **normalized rates** (`missing_tool_rate` = mean of
  `|missing|/|gold|` per row, `extra_tool_rate` = mean of `|extra|/
  |predicted|` per row) — the counts say "how many tools on average," the
  rates say "what fraction of what you needed/predicted was wrong,"
  independent of how many tools a given row happens to need.

Stage 3's periodic validation (`evaluate_policy_on_val`, every
`EVAL_EVERY=200` steps) tracks this same detailed suite — not just the
3-number reward proxy — computed from the same greedy-decoded completions
already generated for the reward, at no extra generation cost.

## Graph-conditioning ablation

`eval/graph_conditioning_ablation.py` runs the **actual** trained Stage
1→2/3 checkpoints twice per test example — once with the correct graph,
once with a different (different-machine) example's graph substituted,
holding the strategy/explanation text fixed — and reports how much Step/MCP/
explanation predictions differ between the two conditions, plus which
condition scores closer to gold. This is distinct from
`graph_adapter_experiments/eval_graph_ablation.py` /
`eval_right_vs_wrong_graph.py`, which are a separate, smaller standalone
proof-of-concept (their own tiny GINE, their own Qwen2.5-1.5B, never touch
`STAGE1_CKPT`/`STAGE2_ADAPTER_DIR`/`STAGE3_ADAPTER_DIR`) — useful evidence
that the soft-prompt mechanism can carry information in principle, not
evidence about whether the real trained model actually uses it.

## Target numbers (Pen-Strategist Step Model, Table 3)

| Metric | Paper | Current best validated (Round 2, single split) | Target |
|---|---|---|---|
| Step accuracy | 82.87% | 76.49% test / 80.75% val | > 82.87% |
| Step Micro-F1 (paper's samples-F1 def.) | 0.80 | — | > 0.80 |
| MCP accuracy (subset) | 48.88% | 52.2% test (fixed-ASL run) | already ahead |
| MCP "Micro F1" (paper's samples-F1 def.) | 0.64 | 0.6944 test (sklearn micro — re-check against `mcp_samples_f1` on the next run) | > 0.64 |

Round 3 (Stage 1 capacity/SupCon/decoupled-retrain changes) and this
session's Stage 2/3 fixes are not yet validated by a real training run — see
`STAGE1_IMPROVEMENTS.md` and `STAGE2_STAGE3_IMPROVEMENTS.md` for what's
pending.

## References

- Ginige, Marasinghe, Jain, Seneviratne, "Pen-Strategist: A Reasoning
  Framework for Penetration Testing Strategy Formation and Analysis,"
  arXiv:2605.04499 — base task, Step Model target numbers.
- Menon, Jayasumana, Rawat, Jain, Veit, Kumar, "Long-tail learning via logit
  adjustment," ICLR 2021, arXiv:2007.07314 — Stage 1 class imbalance.
- Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020,
  arXiv:2004.11362 — Stage 1 SupCon term.
- Kang et al., "Decoupling Representation and Classifier for Long-Tailed
  Recognition," ICLR 2020, arXiv:1910.09217 — Stage 1 decoupled retraining.
- Kim et al., "Hadamard Product for Low-rank Bilinear Pooling," ICLR 2017,
  arXiv:1610.04325 — Stage 1 fusion interaction term.
- Ridnik et al., "Asymmetric Loss For Multi-Label Classification," ICCV
  2021, arXiv:2009.14119 — Stage 1 MCP loss (opt-in).
- Verma et al., "Manifold Mixup," ICML 2019, arXiv:1806.05236 — Stage 1
  regularization.
- Ye et al., dual-clip PPO (2020) — Stage 3 PPO loss.
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale," 2025, arXiv:2503.14476 — Stage 3 Clip-Higher, dynamic sampling.
- Liu et al., "Dr. GRPO" (token-aggregation-bias fix) — considered, not
  adopted; see rationale above.
- BERTScore (Zhang et al., ICLR 2020) — explanation-evaluation secondary
  signal.
