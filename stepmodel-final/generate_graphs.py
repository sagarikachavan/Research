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
straight from that row's PTT cell. By default this uses the deterministic
rule engine in ptt_parser.py (see build_input_json.py's docstring for the
exact State/Action/Finding classification rule) -- no API key needed, and
the result is fully reproducible. Pass --use-llm to parse with an LLM
instead (requires OPENAI_API_KEY).

Machine names are kept exactly as they appear in the CSV, including case
-- "bashed" and "Bashed" are different entries with independent row
sequences, never merged. A defensive (non-merging) guard still makes sure
two differently-named machines can never accidentally write into the same
output directory: if two distinct machine names would sanitize to the same
folder name, the second one gets a short disambiguating suffix instead of
silently overwriting the first's files.

Usage:
    python generate_graphs.py                     # deterministic (default)
    python generate_graphs.py --use-llm            # LLM-based parsing instead
        (requires: export OPENAI_API_KEY=sk-...  &&  pip install --upgrade openai)
"""

import argparse
import json
import re
import hashlib
import pathlib
from threading import Lock

import pandas as pd

import ptt_parser
from ptt_parser import is_valid_machine_name, to_html

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "processed_graph"


def sanitize_dirname(name: str, max_len: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).strip())
    if len(clean) <= max_len:
        return clean
    h = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
    return f"{clean[:max_len]}__{h}"


class DirnameRegistry:
    """Guarantees distinct machine names never share an output directory,
    even if their sanitized names happen to collide (e.g. two names that
    only differ by a character sanitize_dirname strips). Never merges two
    different machine names -- only disambiguates the second one."""

    def __init__(self):
        self._by_key = {}  # sanitized-name.lower() -> (dirname_used, machine_name)
        self._lock = Lock()

    def resolve(self, machine, split_dir):
        base = sanitize_dirname(machine)
        key = base.lower()
        with self._lock:
            existing = self._by_key.get(key)
            if existing is None:
                self._by_key[key] = (base, machine)
                dirname = base
            elif existing[1] == machine:
                dirname = existing[0]
            else:
                h = hashlib.sha1(machine.encode("utf-8")).hexdigest()[:6]
                dirname = f"{base}__{h}"
                self._by_key[f"{key}__{h}"] = (dirname, machine)
        machine_dir = split_dir / dirname
        machine_dir.mkdir(parents=True, exist_ok=True)
        return machine_dir


def _collect_rows(csv_path: pathlib.Path, source_csv_name: str):
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


def _write_graph(graph, machine_dir, row_index):
    fname_base = f"row_{row_index:04d}_graph"
    with open(machine_dir / f"{fname_base}.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    with open(machine_dir / f"{fname_base}.html", "w", encoding="utf-8") as f:
        f.write(to_html(graph))


def process_csv_deterministic(csv_path: pathlib.Path, split: str):
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped_rows, n_machines = _collect_rows(csv_path, csv_path.name)
    registry = DirnameRegistry()

    for i, entry in enumerate(rows, 1):
        graph = ptt_parser.build_row_graph(
            machine=entry["machine"], row_index=entry["row_index"], ptt_text=entry["ptt_text"],
            extra_meta={"csv_row_index": entry["csv_row_index"], "source_csv": entry["source_csv"],
                        "source": "deterministic"},
        )
        machine_dir = registry.resolve(entry["machine"], split_dir)
        _write_graph(graph, machine_dir, entry["row_index"])
        if i % 50 == 0 or i == len(rows):
            print(f"  ... {i}/{len(rows)} rows processed", end="\r")
    print()

    print(f"  {split}: wrote {len(rows)} row graphs across {n_machines} machines -> {split_dir}")
    if skipped_rows:
        print(f"  {split}: skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value "
              f"(see _skipped_rows.json)")
        with open(split_dir / "_skipped_rows.json", "w", encoding="utf-8") as f:
            json.dump(skipped_rows, f, indent=2, ensure_ascii=False)


def process_csv_llm(csv_path: pathlib.Path, split: str, client, model: str, workers: int):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from llm_ptt_parser import parse_ptt_items
    from graph_builder import build_graph_from_items

    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped_rows, n_machines = _collect_rows(csv_path, csv_path.name)
    registry = DirnameRegistry()
    sources = {"llm": 0, "llm_cache": 0, "fallback_regex": 0}

    def _build_and_write(entry):
        items, source = parse_ptt_items(entry["machine"], entry["ptt_text"], client, model=model)
        graph = build_graph_from_items(
            entry["machine"], entry["row_index"], items,
            extra_meta={"csv_row_index": entry["csv_row_index"], "source_csv": entry["source_csv"],
                        "llm_source": source},
        )
        machine_dir = registry.resolve(entry["machine"], split_dir)
        _write_graph(graph, machine_dir, entry["row_index"])
        return source

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_and_write, entry) for entry in rows]
        done = 0
        for fut in as_completed(futures):
            source = fut.result()
            sources[source] = sources.get(source, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)} rows processed", end="\r")
    print()
    print(f"  sources: {sources}")

    print(f"  {split}: wrote {len(rows)} row graphs across {n_machines} machines -> {split_dir}")
    if skipped_rows:
        print(f"  {split}: skipped {len(skipped_rows)} row(s) with a corrupted/invalid Machine value "
              f"(see _skipped_rows.json)")
        with open(split_dir / "_skipped_rows.json", "w", encoding="utf-8") as f:
            json.dump(skipped_rows, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-llm", action="store_true",
                         help="parse each PTT cell with an LLM instead of the deterministic rule engine")
    parser.add_argument("--model", default=None, help="OpenAI model for --use-llm (default: gpt-4o-mini)")
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls, --use-llm only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.use_llm:
        from llm_ptt_parser import get_openai_client, DEFAULT_MODEL
        client = get_openai_client()
        model = args.model or DEFAULT_MODEL

        print("=== Processing training_data.csv [LLM] ===")
        process_csv_llm(DATA_DIR / "training_data.csv", "train", client, model, args.workers)
        print("=== Processing test_data.csv [LLM] ===")
        process_csv_llm(DATA_DIR / "test_data.csv", "test", client, model, args.workers)
    else:
        print("=== Processing training_data.csv [deterministic] ===")
        process_csv_deterministic(DATA_DIR / "training_data.csv", "train")
        print("=== Processing test_data.csv [deterministic] ===")
        process_csv_deterministic(DATA_DIR / "test_data.csv", "test")

    print("Done.")


if __name__ == "__main__":
    main()
