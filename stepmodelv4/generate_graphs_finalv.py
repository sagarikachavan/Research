"""
generate_processed_graphs.py
=============================
For manual visual verification: builds

    processed_graph/
        train/
            <machine>/
                row_0000_graph.json
                row_0000_graph.html
                row_0001_graph.json
                row_0001_graph.html
                ...
        test/
            <machine>/
                ...

Every row of training_data.csv / test_data.csv gets its own graph, built
straight from that row's PTT cell (per-row, not aggregated across rows).
"""

import os
import re
import json
import pathlib
import pandas as pd

from ptt_parser import build_row_graph, to_html, is_valid_machine_name

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "processed_graph"


def sanitize_dirname(name: str, max_len: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).strip())
    if len(clean) <= max_len:
        return clean
    # Some CSV rows have malformed/overlong "Machine" values (looks like a
    # PTT fragment leaked into the column). Truncate and disambiguate with
    # a short hash so distinct overlong values don't collide.
    import hashlib
    h = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
    return f"{clean[:max_len]}__{h}"


def process_csv(csv_path: pathlib.Path, split: str):
    df = pd.read_csv(csv_path)
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Track a per-machine row counter so file names are stable/readable.
    machine_row_counter = {}
    total = 0
    skipped_rows = []

    for csv_row_idx, row in df.iterrows():
        machine_raw = row.get("Machine", "")
        machine = str(machine_raw).strip() if isinstance(machine_raw, str) else str(machine_raw)

        if not is_valid_machine_name(machine):
            skipped_rows.append({
                "csv_row_index": int(csv_row_idx),
                "machine_value_preview": machine[:80],
            })
            continue

        machine_dir_name = sanitize_dirname(machine)
        row_num = machine_row_counter.get(machine, 0)
        machine_row_counter[machine] = row_num + 1

        machine_dir = split_dir / machine_dir_name
        machine_dir.mkdir(parents=True, exist_ok=True)

        graph = build_row_graph(
            machine=machine,
            row_index=row_num,
            ptt_text=row.get("PTT", ""),
            extra_meta={
                "csv_row_index": int(csv_row_idx),
                "source_csv": csv_path.name,
            },
        )

        fname_base = f"row_{row_num:04d}_graph"
        with open(machine_dir / f"{fname_base}.json", "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        with open(machine_dir / f"{fname_base}.html", "w", encoding="utf-8") as f:
            f.write(to_html(graph))

        total += 1

    print(f"  {split}: wrote {total} row graphs across {len(machine_row_counter)} machines -> {split_dir}")
    if skipped_rows:
        print(f"  {split}: skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value "
              f"(see _skipped_rows.json)")
        with open(split_dir / "_skipped_rows.json", "w", encoding="utf-8") as f:
            json.dump(skipped_rows, f, indent=2, ensure_ascii=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Processing training_data.csv ===")
    process_csv(DATA_DIR / "training_data.csv", "train")

    print("=== Processing test_data.csv ===")
    process_csv(DATA_DIR / "test_data.csv", "test")

    print("Done.")


if __name__ == "__main__":
    main()