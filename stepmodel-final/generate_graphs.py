"""
generate_graphs.py
===================
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
from that row's PTT cell via the same LLM-parsing + deterministic
graph-assembly pipeline as build_input_json.py (see llm_ptt_parser.py /
graph_builder.py), and sharing the same on-disk LLM response cache
(.llm_cache/) -- so running this script after (or before) build_input_json.py
against the same CSVs does not re-spend any API calls on rows already
parsed.

Usage:
    export OPENAI_API_KEY=sk-...
    pip install --upgrade openai pandas
    python generate_graphs.py
    python generate_graphs.py --model gpt-4o --workers 4
"""

import argparse
import json
import re
import hashlib
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd

from ptt_parser import is_valid_machine_name, to_html
from llm_ptt_parser import parse_ptt_items, get_openai_client, DEFAULT_MODEL
from graph_builder import build_graph_from_items
from machine_utils import build_canonical_machine_map, canonicalize, report_merged_duplicates

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "processed_graph"


def sanitize_dirname(name: str, max_len: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).strip())
    if len(clean) <= max_len:
        return clean
    # A handful of CSV rows have malformed/overlong "Machine" values;
    # truncate and disambiguate with a short hash so distinct overlong
    # values don't collide on disk.
    h = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
    return f"{clean[:max_len]}__{h}"


def _collect_rows(csv_path: pathlib.Path, source_csv_name: str, canon_map):
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

        # Fold case-duplicate spellings ("bashed" / "Bashed") into one
        # canonical name so their rows stay one continuous sequence and
        # never collide on a case-insensitive filesystem (macOS/Windows).
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


def _resolve_machine_dir(machine, split_dir, dirname_registry, lock):
    """Map a (already-canonicalized) machine name to its output directory,
    guaranteeing distinct machines never share a directory even if their
    sanitized names happen to collide case-insensitively -- the same class
    of bug that motivated machine-name canonicalization in the first
    place, just guarded here as a second line of defense for any pair of
    genuinely different names that sanitize_dirname happens to flatten
    into the same string."""
    base = sanitize_dirname(machine)
    key = base.lower()
    with lock:
        existing = dirname_registry.get(key)
        if existing is None:
            dirname_registry[key] = (base, machine)
            dirname = base
        elif existing[1] == machine:
            dirname = existing[0]
        else:
            import hashlib
            h = hashlib.sha1(machine.encode("utf-8")).hexdigest()[:6]
            dirname = f"{base}__{h}"
            dirname_registry[f"{key}__{h}"] = (dirname, machine)
    machine_dir = split_dir / dirname
    machine_dir.mkdir(parents=True, exist_ok=True)
    return machine_dir


def _build_and_write(entry, client, model, split_dir, dirname_registry, lock):
    items, source = parse_ptt_items(entry["machine"], entry["ptt_text"], client, model=model, row_index=entry["row_index"])
    graph = build_graph_from_items(
        entry["machine"], entry["row_index"], items,
        extra_meta={
            "csv_row_index": entry["csv_row_index"],
            "source_csv": entry["source_csv"],
            "llm_source": source,
        },
    )

    machine_dir = _resolve_machine_dir(entry["machine"], split_dir, dirname_registry, lock)

    fname_base = f"row_{entry['row_index']:04d}_graph"
    with open(machine_dir / f"{fname_base}.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    with open(machine_dir / f"{fname_base}.html", "w", encoding="utf-8") as f:
        f.write(to_html(graph))

    return source


def process_csv(csv_path: pathlib.Path, split: str, client, model: str, workers: int, canon_map):
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped_rows, n_machines = _collect_rows(csv_path, csv_path.name, canon_map)
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}
    dirname_registry = {}
    lock = Lock()

    # Use single thread to avoid cache race conditions
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(_build_and_write, entry, client, model, split_dir, dirname_registry, lock)
                   for entry in rows]
        done = 0
        for fut in as_completed(futures):
            source = fut.result()
            sources[source] = sources.get(source, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)} rows processed", end="\r")

    print()
    print(f"  {split}: wrote {len(rows)} row graphs across {n_machines} machines -> {split_dir}")
    print(f"  sources: {sources}")

    if skipped_rows:
        print(f"  {split}: skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value "
              f"(see _skipped_rows.json)")
        with open(split_dir / "_skipped_rows.json", "w", encoding="utf-8") as f:
            json.dump(skipped_rows, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model used to parse each PTT cell")
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls")
    args = parser.parse_args()

    client = get_openai_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = [DATA_DIR / "training_data.csv", DATA_DIR / "test_data.csv"]
    print("Building canonical machine-name map across both CSVs ...")
    canon_map = build_canonical_machine_map(csv_paths)
    report_merged_duplicates(csv_paths, canon_map)

    print("=== Processing training_data.csv ===")
    process_csv(DATA_DIR / "training_data.csv", "train", client, args.model, args.workers, canon_map)

    print("=== Processing test_data.csv ===")
    process_csv(DATA_DIR / "test_data.csv", "test", client, args.model, args.workers, canon_map)

    print("Done.")


if __name__ == "__main__":
    main()
