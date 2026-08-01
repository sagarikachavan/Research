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

The graph is the State/Action/Finding attack graph produced by
ptt_graph.build_row_graph() directly from that row's PTT cell (per-row,
not aggregated across a machine's rows).
"""

import json
import pathlib
import pandas as pd

from ptt_parser_finalv import build_row_graph, is_valid_machine_name

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


def build_records(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    records = []
    skipped_rows = []

    machine_row_counter = {}

    for csv_row_idx, row in df.iterrows():
        machine = safe_str(row.get("Machine", ""))

        if not is_valid_machine_name(machine):
            # Row's columns are misaligned (PTT text leaked into the
            # Machine column upstream) -- the rest of the row is unreliable
            # too, so skip it rather than emit a bad training example.
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        row_num = machine_row_counter.get(machine, 0)
        machine_row_counter[machine] = row_num + 1

        graph = build_row_graph(
            machine=machine,
            row_index=row_num,
            ptt_text=row.get("PTT", ""),
        )

        record = {
            "machine": machine,
            "graph": graph,
        }
        for csv_col, out_key in CSV_TO_OUTPUT.items():
            record[out_key] = safe_str(row.get(csv_col, ""))

        records.append(record)

    return records, skipped_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("training_data.csv", "train.json"),
        ("test_data.csv", "test.json"),
    ]

    for csv_name, out_name in splits:
        csv_path = DATA_DIR / csv_name
        out_path = OUTPUT_DIR / out_name

        print(f"Processing {csv_name} -> {out_path} ...")
        records, skipped_rows = build_records(csv_path)
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