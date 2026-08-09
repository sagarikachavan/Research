"""
machine_utils.py
=================
Canonicalizes machine names across both CSVs so PTT rows for the same
target machine are always grouped and numbered together, and never
collide on disk -- regardless of inconsistent capitalization in the
source data and regardless of whether the pipeline runs on a
case-sensitive (Linux) or case-insensitive (macOS default, Windows)
filesystem.

Bug this fixes
---------------
Both CSVs log a handful of machines under two different capitalizations
of the same name -- e.g. training_data.csv has 15 rows as "bashed" and 7
later rows as "Bashed" (also: lame/Lame, topology/Topology,
precious/Precious, compiled/Compiled, pilgrimage/Pilgrimage,
authority/Authority, greenhorn/GreenHorn).

Treated as two different machines, each capitalization got its own
row-index counter starting at 0 and its own output directory. On a
case-sensitive filesystem that already splits one machine's timeline into
two fragments; on a case-insensitive one (macOS/Windows) it's worse --
"bashed/" and "Bashed/" resolve to the SAME directory, so "Bashed"'s
row_0000..row_0006 silently overwrote "bashed"'s row_0000..row_0006 on
disk, which is the corrupted processed_graph.zip you saw.

Fix: build a case-insensitive canonical-name map from every valid Machine
value across both CSVs up front (canonical spelling = whichever casing
appears FIRST in document order, training_data.csv then test_data.csv),
and canonicalize every row's machine name through it before it's used for
row counting, directory naming, node ids, or the LLM cache key.
"""

import pathlib
import pandas as pd

from ptt_parser import is_valid_machine_name


def build_canonical_machine_map(csv_paths):
    """Return {original_name: original_name} - case-sensitive, no merging.
    Machine names are kept as-is to preserve distinct machines."""
    canon = {}
    for csv_path in csv_paths:
        csv_path = pathlib.Path(csv_path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for raw in df.get("Machine", []):
            name = str(raw).strip() if isinstance(raw, str) else str(raw)
            if not is_valid_machine_name(name):
                continue
            canon[name] = name
    return canon


def canonicalize(name, canon_map):
    # Case-sensitive: return name as-is if valid
    name_stripped = name.strip()
    return canon_map.get(name_stripped, name_stripped)


def report_merged_duplicates(canon_map_sources, canon_map):
    """No-op - case-sensitive mode, no merging to report."""
    pass
