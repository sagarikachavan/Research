"""
Build input/train.json and input/test.json from CSV data + graph JSONs.

Each output record has:
  Machine, Graph, Previous strategy, Previous step, Previous step result,
  New strategy, Strategy explanation, Gold New step, Gold Step explanation,
  Gold MCP_tasks
"""

import csv
import json
import os
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR = BASE_DIR / "input"

# CSV column name → output key mapping
CSV_TO_OUTPUT = {
    "Machine":            "Machine",
    "Previous strategy":  "Previous strategy",
    "Previous step":      "Previous step",
    "Previous step result": "Previous step result",
    "New strategy":       "New strategy",
    "Strategy explanation": "Strategy explanation",
    "New step":           "Gold New step",
    "Step explanation":   "Gold Step explanation",
    "MCP_tasks":          "Gold MCP_tasks",
}


def find_graph_file(machine: str, split: str) -> pathlib.Path | None:
    """
    Look for <machine>_graph.json inside processed_data/<split>/<machine>/.

    Handles common naming mismatches between CSV values and directory names:
    - Case differences (bashed vs Bashed)
    - Spaces vs underscores (Kioptrix Level 1 vs Kioptrix_Level_1)
    - Dots vs underscores (Typhoon 1.02 vs Typhoon_1.02)
    """
    split_dir = PROCESSED_DIR / split
    if not split_dir.exists():
        return None

    # Normalise a name to a lowercase key for comparison
    def normalise(name: str) -> str:
        return name.lower().replace(" ", "_").replace(".", "_")

    machine_norm = normalise(machine)

    # Build a lookup of normalised dir name → actual Path (built once per call is fine for these sizes)
    for entry in split_dir.iterdir():
        if entry.is_dir() and normalise(entry.name) == machine_norm:
            graph_file = entry / f"{entry.name}_graph.json"
            if graph_file.exists():
                return graph_file

    return None


def load_graph(machine: str, split: str) -> dict | str:
    """Return parsed graph JSON, or an error string if not found."""
    path = find_graph_file(machine, split)
    if path is None:
        print(f"  [WARN] Graph not found for machine '{machine}' in split '{split}'")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_records(csv_path: pathlib.Path, split: str) -> list[dict]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            machine = row["Machine"].strip()
            graph = load_graph(machine, split)

            record = {
                "Machine": machine,
                "Graph": graph,
            }
            for csv_col, out_key in CSV_TO_OUTPUT.items():
                if csv_col == "Machine":
                    continue  # already set above
                record[out_key] = row.get(csv_col, "").strip()

            records.append(record)

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("training_data.csv", "train", "train.json"),
        ("test_data.csv",     "test",  "test.json"),
    ]

    for csv_name, split, out_name in splits:
        csv_path = DATA_DIR / csv_name
        out_path = OUTPUT_DIR / out_name

        print(f"Processing {csv_name} → {out_path} ...")
        records = build_records(csv_path, split)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"  Wrote {len(records)} records to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
