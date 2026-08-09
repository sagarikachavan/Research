"""
generate_dataset.py
===================
Script to generate:
1. input/train.json and input/test.json with format: {machine, graph, new strategy, strategy explanation, gold new step, gold step explanation, gold mcp tasks}
2. processed_graph/train and processed_graph/test with JSON and HTML files for visualization
"""

import json
import pathlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from llm_ptt_parser import parse_ptt_items, get_openai_client
from graph_builder import build_graph_from_items
from ptt_parser import to_html
from machine_utils import build_canonical_machine_map, canonicalize, is_valid_machine_name

DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("processed_graph")
INPUT_DIR = pathlib.Path("input")
DEFAULT_MODEL = "gpt-4o-mini"


def _collect_rows(csv_path, source_csv_name, canon_map):
    """Collect rows from CSV, assigning row_index per machine."""
    df = pd.read_csv(csv_path)
    rows, skipped_rows = [], []
    machine_row_counter = {}

    for csv_row_idx, row in df.iterrows():
        machine_raw = row.get("Machine", "")
        machine = str(machine_raw).strip() if isinstance(machine_raw, str) else str(machine_raw)

        if not is_valid_machine_name(machine):
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        # Drop rows where machine name contains 'PTT' (error rows)
        if "PTT" in machine:
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        machine = canonicalize(machine, canon_map)

        row_num = machine_row_counter.get(machine, 0)
        machine_row_counter[machine] = row_num + 1

        rows.append({
            "machine": machine,
            "row_index": row_num,
            "ptt_text": row.get("PTT", ""),
            "csv_row_index": int(csv_row_idx),
            "source_csv": source_csv_name,
        })

    return rows, skipped_rows, len(machine_row_counter)


def _build_graph(entry, client, model):
    """Build graph for a single entry."""
    items, source = parse_ptt_items(
        entry["machine"], entry["ptt_text"], client, model=model, row_index=entry["row_index"]
    )
    graph = build_graph_from_items(
        entry["machine"], entry["row_index"], items,
        extra_meta={
            "csv_row_index": entry["csv_row_index"],
            "source_csv": entry["source_csv"],
            "llm_source": source,
        },
    )
    return graph, source


def process_csv(csv_path, split, client, model, canon_map):
    """Process a CSV file to generate graphs and input JSON."""
    csv_path = pathlib.Path(csv_path)
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped_rows, n_machines = _collect_rows(csv_path, csv_path.name, canon_map)
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}
    dirname_registry = {}
    lock = Lock()

    # Use single thread to avoid cache race conditions
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [
            pool.submit(_build_graph, entry, client, model)
            for entry in rows
        ]
        done = 0
        input_data = []
        
        for entry, fut in zip(rows, as_completed(futures)):
            graph, source = fut.result()
            sources[source] = sources.get(source, 0) + 1
            done += 1
            
            if done % 25 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)} rows processed", end="\r")

            # Write graph JSON and HTML
            machine_dir = split_dir / entry["machine"]
            machine_dir.mkdir(parents=True, exist_ok=True)
            
            graph_file = machine_dir / f"row_{entry['row_index']:04d}_graph.json"
            html_file = machine_dir / f"row_{entry['row_index']:04d}_graph.html"
            
            with open(graph_file, "w") as f:
                json.dump(graph, f, indent=2)
            
            with open(html_file, "w") as f:
                f.write(to_html(graph))

            # Create input JSON entry
            input_entry = {
                "machine": entry["machine"],
                "graph": graph,
                "new_strategy": "",
                "strategy_explanation": "",
                "gold_new_step": "",
                "gold_step_explanation": "",
                "gold_mcp_tasks": ""
            }
            input_data.append(input_entry)

    print(f"\n  {split}: wrote {len(rows)} row graphs across {n_machines} machines -> {split_dir}")
    print(f"  sources: {sources}")
    if skipped_rows:
        print(f"  {split}: skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value")
        skipped_file = split_dir / "_skipped_rows.json"
        with open(skipped_file, "w") as f:
            json.dump(skipped_rows, f, indent=2)

    # Write input JSON
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_file = INPUT_DIR / f"{split}.json"
    with open(input_file, "w") as f:
        json.dump(input_data, f, indent=2)
    print(f"  {split}: wrote {len(input_data)} entries to {input_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate attack graphs and input JSON from CSV data")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    args = parser.parse_args()

    print("Building canonical machine-name map across both CSVs ...")
    canon_map = build_canonical_machine_map([DATA_DIR / "training_data.csv", DATA_DIR / "test_data.csv"])

    client = get_openai_client()

    print("=== Processing training_data.csv ===")
    process_csv(DATA_DIR / "training_data.csv", "train", client, args.model, canon_map)

    print("=== Processing test_data.csv ===")
    process_csv(DATA_DIR / "test_data.csv", "test", client, args.model, canon_map)

    print("Done.")


if __name__ == "__main__":
    main()
