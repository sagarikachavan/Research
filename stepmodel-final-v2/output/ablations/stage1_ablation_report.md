# Stage-1 modality ablation

Runs found: fusion: seeds [1, 2, 42], text_only: seeds [1, 2, 42], graph_only: seeds [1, 2, 42]

## 1. Test metrics per variant (mean ± std over seeds)

| variant | n | Step acc | Step macro-F1 | MCP samples-F1 | MCP macro-F1 | MCP micro-F1 | MCP subset acc |
|---|---|---|---|---|---|---|---|
| fusion | 3 | 0.789 ± 0.017 | 0.691 ± 0.025 | 0.768 ± 0.013 | 0.673 ± 0.015 | 0.747 ± 0.015 | 0.592 ± 0.021 |
| text_only | 3 | 0.796 ± 0.006 | 0.707 ± 0.020 | 0.740 ± 0.030 | 0.623 ± 0.045 | 0.718 ± 0.032 | 0.560 ± 0.033 |
| graph_only | 3 | 0.628 ± 0.016 | 0.484 ± 0.022 | 0.574 ± 0.036 | 0.367 ± 0.040 | 0.567 ± 0.019 | 0.389 ± 0.022 |

## 2. What removing a modality costs (fusion − variant, seed-matched; positive = performance lost)

| removed | seeds | Step acc | Step macro-F1 | MCP samples-F1 | MCP macro-F1 |
|---|---|---|---|---|---|
| graph removed (fusion − text_only) | 3 | -0.007 | -0.016 | +0.028 | +0.050 |
| text removed (fusion − graph_only) | 3 | +0.160 | +0.208 | +0.194 | +0.307 |

**Hypothesis check (point estimates; CIs in section 3):**

- Step is text-driven: removing the text costs more step accuracy (+0.160) than removing the graph (-0.007)  ->  SUPPORTED
- MCP is graph-driven: removing the graph costs more MCP samples-F1 (+0.028) than removing the text (+0.194)  ->  NOT supported
- Double dissociation (both): **NO**

## 3. Paired bootstrap over the test rows (95% CI; * = interval excludes 0)

| comparison | Δ step accuracy [CI] | Δ MCP samples-F1 [CI] |
|---|---|---|
| fusion − text_only  (value of the graph) | -0.007 [-0.029, +0.014] | +0.028 [+0.010, +0.046]* |
| fusion − graph_only (value of the text) | +0.160 [+0.107, +0.215]* | +0.194 [+0.152, +0.235]* |
| text_only − graph_only | +0.168 [+0.114, +0.221]* | +0.166 [+0.122, +0.209]* |

Expected under the hypothesis: row 3 positive on step, negative on MCP; row 1 small on step, large on MCP.
(Models averaged over seeds per variant: fusion n=3, graph_only n=3, text_only n=3; n_boot=5000.)

Plot: /home/dgxuser/Research/stepmodel-final-v2/output/ablations/stage1_modality_ablation.png
