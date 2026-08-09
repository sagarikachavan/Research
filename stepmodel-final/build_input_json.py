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

The graph is built in two steps:
  1. llm_ptt_parser.parse_ptt_items() sends the row's PTT cell to an LLM
     (OpenAI API) and gets back a flat, ordered, classified list of
     State/Action items with status + findings. This replaces the old
     regex-only parser for the ambiguous NLP part of the job (deciding
     what's a phase vs. an action vs. plain context, where status markers
     and findings payloads sit, folding bare nested data blocks into their
     parent). If the LLM call fails validation after retries, a
     deterministic regex fallback is used instead so no row is dropped.
  2. graph_builder.build_graph_from_items() deterministically turns that
     classified list into the vis-network-style node/edge JSON (ids,
     colors, per-status shading, edge wiring, statistics) -- a fixed
     transformation that needs no LLM.

LLM responses are cached in .llm_cache/ keyed by (model, machine, ptt_text),
shared with generate_graphs.py, so running both against the same CSVs only
pays for each unique row once.

Usage:
    export OPENAI_API_KEY=sk-...
    pip install --upgrade openai pandas
    python build_input_json.py                       # full run
    python build_input_json.py --limit 20             # quick smoke test
    python build_input_json.py --model gpt-4o --workers 4
"""

import argparse
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ptt_parser import is_valid_machine_name
from llm_ptt_parser import parse_ptt_items, get_openai_client, DEFAULT_MODEL
from graph_builder import build_graph_from_items
from machine_utils import build_canonical_machine_map, canonicalize, report_merged_duplicates

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


def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _collect_rows(csv_path: pathlib.Path, canon_map, limit=None):
    """Read the CSV and return (rows, skipped_rows). `rows` holds everything
    needed to build one record: machine, per-machine row index, PTT text,
    and the already-extracted gold columns."""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    rows, skipped_rows = [], []
    machine_row_counter = {}

    for csv_row_idx, row in df.iterrows():
        machine = safe_str(row.get("Machine", ""))

        if not is_valid_machine_name(machine):
            # Row's columns are misaligned (PTT text leaked into the
            # Machine column upstream) -- skip rather than emit a bad
            # training example.
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        # Fold case-duplicate spellings ("bashed" / "Bashed") into one
        # canonical name so their rows stay one continuous sequence
        # instead of two separate, colliding ones.
        machine = canonicalize(machine, canon_map)

        row_num = machine_row_counter.get(machine, 0)
        machine_row_counter[machine] = row_num + 1

        entry = {
            "machine": machine,
            "row_index": row_num,
            "ptt_text": row.get("PTT", ""),
            "csv_row_index": int(csv_row_idx),
        }
        for csv_col, out_key in CSV_TO_OUTPUT.items():
            entry[out_key] = safe_str(row.get(csv_col, ""))
        rows.append(entry)

    return rows, skipped_rows


def _build_one_record(entry, client, model):
    items, source = parse_ptt_items(entry["machine"], entry["ptt_text"], client, model=model)
    graph = build_graph_from_items(
        entry["machine"], entry["row_index"], items,
        extra_meta={"csv_row_index": entry["csv_row_index"], "llm_source": source},
    )
    record = {"machine": entry["machine"], "graph": graph}
    for out_key in CSV_TO_OUTPUT.values():
        record[out_key] = entry[out_key]
    return record, source


def build_records(csv_path: pathlib.Path, client, model, workers, canon_map, limit=None):
    rows, skipped_rows = _collect_rows(csv_path, canon_map, limit=limit)
    records = [None] * len(rows)
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one_record, entry, client, model): i
                   for i, entry in enumerate(rows)}
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model used to parse each PTT cell")
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N rows per CSV (for a quick/cheap smoke test)")
    parser.add_argument("--no-cache", action="store_true", help="ignore/skip the on-disk LLM response cache")
    args = parser.parse_args()

    client = get_openai_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("training_data.csv", "train.json"),
        ("test_data.csv", "test.json"),
    ]
    csv_paths = [DATA_DIR / name for name, _ in splits]

    print("Building canonical machine-name map across both CSVs ...")
    canon_map = build_canonical_machine_map(csv_paths)
    report_merged_duplicates(csv_paths, canon_map)

    for csv_name, out_name in splits:
        csv_path = DATA_DIR / csv_name
        out_path = OUTPUT_DIR / out_name

        print(f"Processing {csv_name} -> {out_path} (model={args.model}, workers={args.workers}) ...")
        records, skipped_rows = build_records(csv_path, client, args.model, args.workers,
                                               canon_map, limit=args.limit)

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
