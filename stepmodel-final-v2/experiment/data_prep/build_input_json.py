"""
build_input_json.py
====================
Build input/train.json and input/test.json for text-only experiment.

Each record (one per CSV row) has:
    {
      "machine": "...",
      "new_strategy": "...",
      "strategy_explanation": "...",
      "gold_new_step": "...",
      "gold_step_explanation": "...",
      "gold_mcp_tasks": "..."
    }

This is a simplified version that does NOT use graph information.
Only text fields are used: new_strategy and strategy_explanation as input,
and gold_new_step, gold_step_explanation, gold_mcp_tasks as targets.

Usage:
    python build_input_json.py
    python build_input_json.py --limit 20
"""

import argparse
import json
import pathlib

import pandas as pd

BASE_DIR = pathlib.Path(__file__).parent.parent
# Use main pipeline's data directory
MAIN_ROOT = BASE_DIR.parent
DATA_DIR = MAIN_ROOT / "data"
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


def build_records(csv_path: pathlib.Path, limit=None):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    records = []
    skipped_rows = []

    for csv_row_idx, row in df.iterrows():
        machine = safe_str(row.get("Machine", ""))

        if not machine:
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        record = {"machine": machine}
        for csv_col, out_key in CSV_TO_OUTPUT.items():
            record[out_key] = safe_str(row.get(csv_col, ""))
        records.append(record)

    return records, skipped_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N rows per CSV (for a quick check)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("training_data.csv", "train.json"),
        ("test_data.csv", "test.json"),
    ]

    for csv_name, out_name in splits:
        csv_path = DATA_DIR / csv_name
        out_path = OUTPUT_DIR / out_name

        print(f"Processing {csv_name} -> {out_path} ...")
        records, skipped_rows = build_records(csv_path, limit=args.limit)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {len(records)} records to {out_path}")

        if skipped_rows:
            skip_path = OUTPUT_DIR / f"_skipped_rows_{out_name}"
            print(f"  Skipped {len(skipped_rows)} row(s) with empty Machine value "
                  f"(see {skip_path.name})")
            with open(skip_path, "w", encoding="utf-8") as f:
                json.dump(skipped_rows, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
