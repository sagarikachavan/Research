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

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

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

# INPUT CONTRACT: each record's model input is machine + graph +
# new_strategy + strategy_explanation ONLY. No other fields are carried
# into the model input. (A previous version of this file also carried
# forward "previous_strategy"/"previous_step"/"previous_step_result" from
# the same machine's prior row -- removed: it wasn't part of the requested
# input schema, and "previous_step" duplicated the gold "New step" label
# text of the prior row, letting the model partly solve step classification
# by copying step-to-step transition frequency instead of reasoning over
# the graph + strategy.)
EXTRA_OUTPUT_KEYS = []


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


def _items_from_ptt(ptt_text):
    parsed = ptt_parser.parse_ptt(ptt_text)
    return [{"number": it["number"], "title": it["title"],
              "type": "State" if ptt_parser.classify(it) == "state" else "Action",
              "status": it["status"], "payload": it["payload"]} for it in parsed]


def _build_one_record_deterministic(entry):
    from graph_builder import validate_row_graph

    graph = ptt_parser.build_row_graph(
        machine=entry["machine"],
        row_index=entry["row_index"],
        ptt_text=entry["ptt_text"],
        extra_meta={"csv_row_index": entry["csv_row_index"], "source": "deterministic"},
    )
    problems = validate_row_graph(_items_from_ptt(entry["ptt_text"]), graph,
                                   entry["machine"], entry["row_index"])
    record = {"machine": entry["machine"], "graph": graph}
    for out_key in CSV_TO_OUTPUT.values():
        record[out_key] = entry[out_key]
    for out_key in EXTRA_OUTPUT_KEYS:
        record[out_key] = entry[out_key]
    return record, problems


def build_records_deterministic(csv_path: pathlib.Path, limit=None):
    rows, skipped_rows = _collect_rows(csv_path, limit=limit)
    records, row_problems = [], []
    for entry in rows:
        record, problems = _build_one_record_deterministic(entry)
        records.append(record)
        if problems:
            row_problems.append({"machine": entry["machine"], "row_index": entry["row_index"],
                                  "problems": problems})
    return records, skipped_rows, row_problems, []


def build_records_llm(csv_path: pathlib.Path, client, model, workers, limit=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from llm_ptt_parser import parse_ptt_items
    from graph_builder import build_graph_from_items, validate_row_graph

    rows, skipped_rows = _collect_rows(csv_path, limit=limit)
    records = [None] * len(rows)
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}
    row_problems = []

    def _build(entry):
        items, source = parse_ptt_items(entry["machine"], entry["ptt_text"], client, model=model)
        graph = build_graph_from_items(
            entry["machine"], entry["row_index"], items,
            extra_meta={"csv_row_index": entry["csv_row_index"], "llm_source": source},
        )
        problems = validate_row_graph(items, graph, entry["machine"], entry["row_index"])
        record = {"machine": entry["machine"], "graph": graph}
        for out_key in CSV_TO_OUTPUT.values():
            record[out_key] = entry[out_key]
        for out_key in EXTRA_OUTPUT_KEYS:
            record[out_key] = entry[out_key]
        return record, source, problems

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build, entry): i for i, entry in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            record, source, problems = fut.result()
            records[i] = record
            sources[source] = sources.get(source, 0) + 1
            if problems:
                row_problems.append({"machine": rows[i]["machine"], "row_index": rows[i]["row_index"],
                                      "problems": problems})
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)} rows processed", end="\r")
    print()
    print(f"  sources: {sources}")
    return records, skipped_rows, row_problems, []


def build_records_hybrid(csv_path: pathlib.Path, client, model, workers, limit=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from llm_ptt_parser import parse_ptt_items_hybrid
    from graph_builder import build_graph_from_items, validate_row_graph

    rows, skipped_rows = _collect_rows(csv_path, limit=limit)
    records = [None] * len(rows)
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}
    row_problems, all_disagreements = [], []
    lock = Lock()

    def _build(entry):
        disagreement_log = []
        items, source = parse_ptt_items_hybrid(
            entry["machine"], entry["ptt_text"], client, model=model,
            disagreement_log=disagreement_log,
        )
        graph = build_graph_from_items(
            entry["machine"], entry["row_index"], items,
            extra_meta={"csv_row_index": entry["csv_row_index"], "source": "hybrid", "llm_source": source},
        )
        problems = validate_row_graph(items, graph, entry["machine"], entry["row_index"])
        record = {"machine": entry["machine"], "graph": graph}
        for out_key in CSV_TO_OUTPUT.values():
            record[out_key] = entry[out_key]
        for out_key in EXTRA_OUTPUT_KEYS:
            record[out_key] = entry[out_key]
        return record, source, problems, disagreement_log

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build, entry): i for i, entry in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            record, source, problems, disagreement_log = fut.result()
            records[i] = record
            with lock:
                sources[source] = sources.get(source, 0) + 1
                if problems:
                    row_problems.append({"machine": rows[i]["machine"], "row_index": rows[i]["row_index"],
                                          "problems": problems})
                all_disagreements.extend(disagreement_log)
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)} rows processed", end="\r")
    print()
    print(f"  sources: {sources}")
    return records, skipped_rows, row_problems, all_disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rule", "llm", "hybrid"], default="rule",
                         help="rule = deterministic only (default, no API key needed); "
                              "llm = LLM parses structure + classification from scratch; "
                              "hybrid = deterministic structure + LLM classifies only the "
                              "ambiguous items (recommended -- best accuracy, fewest tokens)")
    parser.add_argument("--use-llm", action="store_true", help="deprecated alias for --mode llm")
    parser.add_argument("--model", default=None, help="OpenAI model for --mode llm/hybrid (default: gpt-4o-mini)")
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls, --mode llm/hybrid only")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N rows per CSV (for a quick check)")
    args = parser.parse_args()
    mode = "llm" if args.use_llm else args.mode

    client = None
    model = None
    if mode in ("llm", "hybrid"):
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

        mode_desc = {"rule": "deterministic rule engine",
                     "llm": f"LLM (model={model}, workers={args.workers})",
                     "hybrid": f"hybrid: rule structure + LLM classify (model={model}, workers={args.workers})"}[mode]
        print(f"Processing {csv_name} -> {out_path}  [{mode_desc}] ...")

        if mode == "llm":
            records, skipped_rows, row_problems, disagreements = build_records_llm(
                csv_path, client, model, args.workers, limit=args.limit)
        elif mode == "hybrid":
            records, skipped_rows, row_problems, disagreements = build_records_hybrid(
                csv_path, client, model, args.workers, limit=args.limit)
        else:
            records, skipped_rows, row_problems, disagreements = build_records_deterministic(
                csv_path, limit=args.limit)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {len(records)} records to {out_path}")

        if skipped_rows:
            skip_path = OUTPUT_DIR / f"_skipped_rows_{out_name}"
            print(f"  Skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value "
                  f"(see {skip_path.name})")
            with open(skip_path, "w", encoding="utf-8") as f:
                json.dump(skipped_rows, f, indent=2, ensure_ascii=False)

        if row_problems:
            report_path = OUTPUT_DIR / f"_validation_report_{out_name}"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(row_problems, f, indent=2, ensure_ascii=False)
            n_rows = len({(p["machine"], p["row_index"]) for p in row_problems})
            print(f"  *** {len(row_problems)} validation problem(s) across {n_rows} row(s) -- "
                  f"see {report_path.name} ***")
        else:
            print(f"  validation clean -- every row passed all structural + hard-rule checks")

        if disagreements:
            dis_path = OUTPUT_DIR / f"_llm_disagreements_{out_name}"
            with open(dis_path, "w", encoding="utf-8") as f:
                json.dump(disagreements, f, indent=2, ensure_ascii=False)
            print(f"  LLM overrode the deterministic classification on {len(disagreements)} item(s) "
                  f"-- see {dis_path.name} for the full audit trail")

    print("Done.")


if __name__ == "__main__":
    main()