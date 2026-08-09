# PTT -> Attack Graph pipeline (LLM-based)

## What changed

The old pipeline parsed each PTT cell with a hand-written regex
(`ptt_parser.parse_ptt` + `classify`). That's brittle — status markers show
up before/after the payload or not at all, some items are bare `{...}` data
blocks with no title, and deciding "is this a State or an Action" (e.g. a
Target IP with a Findings payload is a State, not an Action) is a judgement
call, not a pattern match.

The new pipeline splits the job in two:

1. **`llm_ptt_parser.py`** — sends each row's raw PTT text to an OpenAI
   model and gets back a flat, ordered list of classified items
   (`number`, `title`, `node_type`, `status`, `finding`). This is the part
   that actually needs language understanding, so it's delegated to the
   LLM instead of regex. If a call fails validation after 3 retries, it
   falls back to the original regex parser so **no row is ever dropped**
   — you'll see it recorded as `"llm_source": "fallback_regex"` in the
   output graph.
2. **`graph_builder.py`** — deterministically turns that classified list
   into the same vis-network node/edge JSON schema as before (ids, colors,
   status shading, the four edge types). This part is plain Python, costs
   nothing, and always produces the same graph for the same item list.

`build_input_json.py` and `generate_graphs.py` are both thin drivers over
this shared pipeline — same logic, same schema as the graphs you already
have in `processed_graph.zip`.

## Setup

```bash
pip install --upgrade openai pandas
export OPENAI_API_KEY=sk-...          # your OpenAI token
```

Put `training_data.csv` and `test_data.csv` in a `data/` folder next to
these scripts (same layout as before).

## Run

```bash
# Quick, cheap smoke test on the first 20 rows of each CSV
python build_input_json.py --limit 20

# Full run
python build_input_json.py

# Visual verification folder (same rows, shares the cache below)
python generate_graphs.py
```

Optional flags on both scripts:
- `--model gpt-4o-mini` (default) or any other OpenAI chat model that
  supports structured JSON outputs, e.g. `--model gpt-4o`
- `--workers 8` (default) — number of PTT cells parsed concurrently

## Caching

Every LLM response is cached on disk under `.llm_cache/`, keyed by a hash
of `(model, machine, ptt_text)`. Since `build_input_json.py` and
`generate_graphs.py` process the exact same CSV rows, **running the second
script after the first costs zero additional API calls** — it just reads
the cache. Delete `.llm_cache/` (or pass a different `--model`) to force
re-parsing.

## Cost / scale note

`training_data.csv` + `test_data.csv` together are ~2,200 rows, so a full
run is ~2,200 LLM calls the first time (then free on any re-run thanks to
the cache). With `gpt-4o-mini` and `--workers 8` this should be inexpensive
and finish in well under an hour; use `--limit` first to sanity-check the
graphs before committing to the full run.

## Files

| File | Purpose |
|---|---|
| `ptt_parser.py` | Unchanged. Color/shade constants, `is_valid_machine_name`, and the original regex parser (used only as the fallback path). |
| `llm_ptt_parser.py` | LLM call, JSON-schema validation, disk cache, regex fallback. |
| `graph_builder.py` | Deterministic item-list -> node/edge graph JSON assembly (+ same HTML via `ptt_parser.to_html`). |
| `machine_utils.py` | Canonicalizes case-duplicate machine names (see "Known data issue" below). |
| `build_input_json.py` | Produces `input/train.json`, `input/test.json`. |
| `generate_graphs.py` | Produces `processed_graph/{train,test}/<machine>/row_NNNN_graph.{json,html}` for manual visual QA. |

## Known data issue: case-duplicate machine names (fixed)

Both CSVs log 8 machines under two different capitalizations of the same
name (`bashed`/`Bashed`, `lame`/`Lame`, `topology`/`Topology`,
`precious`/`Precious`, `compiled`/`Compiled`, `pilgrimage`/`Pilgrimage`,
`authority`/`Authority`, `greenhorn`/`GreenHorn`). Left alone, each
capitalization got its own row-index counter starting at 0 and its own
output folder — which, if you generate on a case-insensitive filesystem
(macOS default, Windows), makes `bashed/` and `Bashed/` resolve to the
*same* folder, so the second variant's `row_0000...` files silently
overwrote the first's. That's the corrupted `processed_graph.zip` you saw.

Fixed in `machine_utils.py`: before processing either CSV, both scripts
now scan **both** CSVs and build a case-insensitive canonical name map
(canonical spelling = whichever casing appears first in the data), then
canonicalize every row's machine name through it. All 8 duplicates now
merge into one continuous sequence per machine. Both scripts print which
names got merged, e.g.:

```
Merged 8 case-duplicate machine name(s) so their rows stay one continuous sequence:
  ['Bashed', 'bashed'] -> "bashed"
  ...
```

`generate_graphs.py` also keeps a second, independent safety net
(`_resolve_machine_dir`) that disambiguates any *other* pair of distinct
machine names whose sanitized folder names happen to collide — so this
class of bug can't recur even for machine names not on the list above.
