# PTT -> Attack Graph pipeline

> **Note:** this repo was restructured — see `CHANGES_AND_FINDINGS.md` in the
> repo root for the full audit (pipeline sanity check, leakage check, the
> Stage 3 checkpoint-selection fix, files removed, and the new folder
> layout: `core/` `data_prep/` `training/` `eval/`). Everything below still
> runs the same way, just from `data_prep/build_input_json.py` etc. instead
> of the repo root.

## Default mode: deterministic (no API key needed)

`build_input_json.py` and `generate_graphs.py` now build graphs with a
**deterministic rule engine** (`ptt_parser.py`) by default -- no LLM, no
API key, fully reproducible, and it implements your exact classification
rule:

1. **No finding/data payload attached -> always State.** Even if the
   title reads like a command (e.g. `2.1 Identify vulnerability -
   (to-do)`), nothing has been produced yet, so it's just the current
   phase/sub-phase.
2. **Has a payload, but is contextual/informational** (a machine name,
   target IP, "Authentication Status", "Host Info", ...) **-> State**,
   with the payload kept on that State node itself (no separate Finding
   node).
3. **Has a payload and describes a concrete action** ("Perform a port
   scan", "Enumerate HTTP service", "Check user privileges", ...)
   **-> Action**, and the payload becomes a separate Finding node.

Every numbered item -- **including a Target IP / IP-address item that has
its own number** (e.g. `1.4 Target IP: {Findings: 10.129.X.X}`,
`1.9.1 IP Address - {...}`) -- gets its own node. Nothing is folded into a
sibling or parent unless it's a truly bare `{...}` block with no label of
its own, directly under a payload-less parent -- which is rare in this
dataset and, when it happens, is verified item-by-item, not guessed at.

I verified this against the exact case you flagged (`active`, row 9,
`1.4 Target IP`, `1.9.1 IP Address`, etc.) and against the full dataset
(1,996 kept rows across both CSVs): every Action node pairs with exactly
one Finding node, and Target-IP-style items with their own PTT number
always land as their own State node.

An **LLM mode** (`--use-llm`, needs `OPENAI_API_KEY`) is still available
if you want it for genuinely ambiguous wording -- its prompt now encodes
the identical decision rule above -- but it's opt-in, not the default,
since deterministic parsing is what actually matched your spec when
checked against real rows.

## What was wrong before (fixed)

1. **Machine names were being merged across case (`bashed`/`Bashed`).**
   You confirmed these are *not* duplicates -- each is a separate entry.
   Reverted: machine names are now kept exactly as they appear in the
   CSV, case and all. Row numbering is per exact machine string, just
   like your original data. A defensive (non-merging) guard still stops
   two *different* machine names from ever colliding on disk -- if two
   distinct names would sanitize to the same folder, the second gets a
   short disambiguating suffix instead of silently overwriting the first.
2. **Target IP items with their own PTT number were sometimes folded
   away instead of getting their own State node** (the `active`/row 9
   case you flagged). Root cause: the LLM prompt's "fold bare children"
   instruction was too loose and occasionally swallowed a legitimately
   separate item. Fixed two ways: (a) the new deterministic engine only
   folds a child when it is *truly* label-less and one level below a
   payload-less parent, verified structurally, not guessed; (b) the LLM
   prompt (for `--use-llm` mode) was tightened to the same rule with an
   explicit "when in doubt, do NOT fold" instruction.
3. **A handful of noun-phrase field labels that happen to share a word
   root with an action verb** (e.g. "Authentication Status" contains
   "authenticat...") were misclassified as Action. Fixed with an
   `INFO_SUFFIX_RE` check in `ptt_parser.classify()` that recognizes
   common trailing field-label nouns (Status, Info, Address, Sessions,
   Resources, Duration, ...) and always treats them as State, checked
   before the action-verb search.
4. **Rows with a corrupted "Machine" column** (PTT text leaked into it
   upstream -- e.g. `"1. Reconnaissance {to-do}\n1.1 Port scanning ..."`
   sitting in the Machine cell) are dropped, not guessed at. This was
   already handled by `ptt_parser.is_valid_machine_name` and is unchanged;
   confirmed it drops exactly the rows with this problem (166 in
   `training_data.csv`, 0 in `test_data.csv`) and nothing else.

## Setup

```bash
pip install --upgrade pandas
```

Put `training_data.csv` and `test_data.csv` in a `data/` folder next to
these scripts.

Only needed for `--use-llm`:
```bash
pip install --upgrade openai
export OPENAI_API_KEY=sk-...
```

## Run

```bash
# Full dataset, deterministic (default) -- fast, free, reproducible
python build_input_json.py
python generate_graphs.py

# Quick check on a handful of rows first
python build_input_json.py --limit 20

# LLM mode instead
python build_input_json.py --use-llm
python generate_graphs.py --use-llm
```

## Files

| File | Purpose |
|---|---|
| `ptt_parser.py` | Core deterministic engine: PTT text -> classified items -> node/edge attack graph JSON + HTML (`to_html`). Also `is_valid_machine_name` (drops corrupted rows). |
| `llm_ptt_parser.py` | LLM call, JSON-schema validation, disk cache, regex fallback -- used only in `--use-llm` mode. |
| `graph_builder.py` | Deterministic item-list -> node/edge graph assembly, used by `--use-llm` mode (LLM only classifies items; graph construction itself stays deterministic either way). |
| `build_input_json.py` | Produces `input/train.json`, `input/test.json`. |
| `generate_graphs.py` | Produces `processed_graph/{train,test}/<machine>/row_NNNN_graph.{json,html}` for manual visual QA. |

## If you spot more mismatches

The classification rule above is regex/heuristic-based, so it won't be
100% perfect on every one of the ~2,000 rows' worth of free-text titles.
If you find another specific row where the graph still doesn't match the
PTT, send me the exact `machine` + row number (or the PTT text itself) and
I'll trace it through `ptt_parser.classify()` the same way I did for
`active` row 9, and patch the specific rule that's misfiring.
