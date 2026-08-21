"""
build_input_json.py
====================
Build input/train.json and input/test.json.

Each record (one per CSV row) has:
    {
      "machine": "...",
      "graph": <attack graph JSON built from that row's PTT>,
      "new_strategy": "...",
      "strategy_explanation": "...",
      "gold_new_step": "...",
      "gold_step_explanation": "...",
      "gold_mcp_tasks": "..."
    }

Graph construction, by default, is fully deterministic
(ptt_parser.build_row_graph): every numbered PTT item becomes its own node
-- classified State vs Action using the exact rule you specified:

    * No finding/data payload attached  -> always State (nothing produced
      yet, it's just the current phase/sub-phase).
    * Has a payload AND reads like a concrete action ("Perform a port
      scan", "Enumerate HTTP service", ...)            -> Action, payload
      becomes a separate Finding node.
    * Has a payload but is really contextual/informational (a machine
      name, IP address, "Authentication Status", ...)  -> State, with the
      payload kept on that State node itself (no separate Finding node).

Every numbered item -- including a "Target IP"/IP-address item that has
its own number (e.g. 1.4, 1.9.1) -- gets its own node. Nothing is folded
into a sibling or parent unless it is a truly bare "{...}" data block with
no label of its own, sitting exactly one level under a payload-less
parent.

This is deterministic and needs no API key, so it's the default. If you'd
rather have an LLM parse each PTT cell (more flexible on genuinely
ambiguous wording, but can drift from the exact rule above -- see
llm_ptt_parser.py's SYSTEM_PROMPT), pass --use-llm.

Rows whose "Machine" column is corrupted (PTT text leaked into it upstream)
are dropped, not guessed at -- see ptt_parser.is_valid_machine_name.

Machine names are kept exactly as they appear in the CSV, including case
("bashed" and "Bashed" are treated as different entries, each with its own
independent row sequence) -- only file/directory naming is guarded against
accidental collisions (see generate_graphs.py), never merged.

Usage:
    python build_input_json.py                          # deterministic (default)
    python build_input_json.py --limit 20                # quick check
    python build_input_json.py --use-llm                 # LLM-based parsing instead
        (requires: export OPENAI_API_KEY=sk-...  &&  pip install --upgrade openai)
"""

import argparse
import json
import pathlib

import pandas as pd

import ptt_parser
from ptt_parser import is_valid_machine_name

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "input"

# CSV column -> output key
CSV_TO_OUTPUT = {
    "New strategy": "new_strategy",
    "Strategy explanation": "strategy_explanation",
    "New step": "gold_new_step",
    "Step explanation": "gold_step_explanation",
    "MCP_tasks": "gold_mcp_tasks",
}

# Synthesized "what happened last turn" context fields. These are NOT read
# from a CSV column -- they're carried forward from the SAME machine's PRIOR
# row in _collect_rows() below. This intentionally revives the idea behind
# an earlier "Previous strategy"/"Previous step"/"Previous step result"
# context (removed from this codebase because it leaked: it had been
# populated with the CURRENT row's own new_strategy/gold_new_step/
# gold_step_explanation, i.e. the label predicting itself). Done correctly,
# this is legitimate sequential context -- the actual previous step for that
# machine -- and carries no information about the current row's own label.
EXTRA_OUTPUT_KEYS = ["previous_strategy", "previous_step", "previous_step_result"]


def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _collect_rows(csv_path: pathlib.Path, limit=None):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    rows, skipped_rows = [], []
    machine_row_counter = {}
    # Most recently seen (new_strategy, gold_new_step, gold_step_explanation)
    # per machine, used to populate the NEXT row's previous_* context. Never
    # read from within the same iteration that produces a row's own labels.
    machine_last_step = {}

    for csv_row_idx, row in df.iterrows():
        machine = safe_str(row.get("Machine", ""))

        if not is_valid_machine_name(machine):
            # Row's columns are misaligned (PTT text leaked into the
            # Machine column upstream) -- the rest of the row is unreliable
            # too, so drop it rather than emit a bad training example.
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        row_num = machine_row_counter.get(machine, 0)
        machine_row_counter[machine] = row_num + 1

        prev = machine_last_step.get(machine)
        entry = {
            "machine": machine,
            "row_index": row_num,
            "ptt_text": row.get("PTT", ""),
            "csv_row_index": int(csv_row_idx),
            "previous_strategy":     prev["new_strategy"] if prev else "",
            "previous_step":         prev["gold_new_step"] if prev else "",
            "previous_step_result":  prev["gold_step_explanation"] if prev else "",
        }
        for csv_col, out_key in CSV_TO_OUTPUT.items():
            entry[out_key] = safe_str(row.get(csv_col, ""))
        rows.append(entry)

        # This row's own labels become the "previous" context for the NEXT
        # row of the same machine -- recorded only after entry is built, so
        # it can never leak into the row that produced it.
        machine_last_step[machine] = {
            "new_strategy":         entry["new_strategy"],
            "gold_new_step":        entry["gold_new_step"],
            "gold_step_explanation": entry["gold_step_explanation"],
        }

    return rows, skipped_rows


def _build_one_record_deterministic(entry):
    graph = ptt_parser.build_row_graph(
        machine=entry["machine"],
        row_index=entry["row_index"],
        ptt_text=entry["ptt_text"],
        extra_meta={"csv_row_index": entry["csv_row_index"], "source": "deterministic"},
    )
    record = {"machine": entry["machine"], "graph": graph}
    for out_key in CSV_TO_OUTPUT.values():
        record[out_key] = entry[out_key]
    for out_key in EXTRA_OUTPUT_KEYS:
        record[out_key] = entry[out_key]
    return record


def build_records_deterministic(csv_path: pathlib.Path, limit=None):
    rows, skipped_rows = _collect_rows(csv_path, limit=limit)
    records = [_build_one_record_deterministic(entry) for entry in rows]
    return records, skipped_rows


def build_records_llm(csv_path: pathlib.Path, client, model, workers, limit=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from llm_ptt_parser import parse_ptt_items
    from graph_builder import build_graph_from_items

    rows, skipped_rows = _collect_rows(csv_path, limit=limit)
    records = [None] * len(rows)
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}

    def _build(entry):
        items, source = parse_ptt_items(entry["machine"], entry["ptt_text"], client, model=model)
        graph = build_graph_from_items(
            entry["machine"], entry["row_index"], items,
            extra_meta={"csv_row_index": entry["csv_row_index"], "llm_source": source},
        )
        record = {"machine": entry["machine"], "graph": graph}
        for out_key in CSV_TO_OUTPUT.values():
            record[out_key] = entry[out_key]
        for out_key in EXTRA_OUTPUT_KEYS:
            record[out_key] = entry[out_key]
        return record, source

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build, entry): i for i, entry in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            record, source = fut.result()
            records[i] = record
            sources[source] = sources.get(source, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)} rows processed", end="\r")
    print()
    print(f"  sources: {sources}")
    return records, skipped_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-llm", action="store_true",
                         help="parse each PTT cell with an LLM instead of the deterministic rule engine")
    parser.add_argument("--model", default=None, help="OpenAI model for --use-llm (default: gpt-4o-mini)")
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls, --use-llm only")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N rows per CSV (for a quick check)")
    args = parser.parse_args()

    client = None
    model = None
    if args.use_llm:
        from llm_ptt_parser import get_openai_client, DEFAULT_MODEL
        client = get_openai_client()
        model = args.model or DEFAULT_MODEL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("training_data.csv", "train.json"),
        ("test_data.csv", "test.json"),
    ]

    for csv_name, out_name in splits:
        csv_path = DATA_DIR / csv_name
        out_path = OUTPUT_DIR / out_name

        mode = f"LLM (model={model}, workers={args.workers})" if args.use_llm else "deterministic rule engine"
        print(f"Processing {csv_name} -> {out_path}  [{mode}] ...")

        if args.use_llm:
            records, skipped_rows = build_records_llm(csv_path, client, model, args.workers, limit=args.limit)
        else:
            records, skipped_rows = build_records_deterministic(csv_path, limit=args.limit)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {len(records)} records to {out_path}")

        if skipped_rows:
            skip_path = OUTPUT_DIR / f"_skipped_rows_{out_name}"
            print(f"  Skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value "
                  f"(see {skip_path.name})")
            with open(skip_path, "w", encoding="utf-8") as f:
                json.dump(skipped_rows, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()