"""
graph_builder.py
=================
Deterministic assembly of a State/Action/Finding attack graph (vis-network
compatible node/edge JSON) from an already-classified, ordered list of PTT
items.

This is intentionally split out from *parsing*. Turning free-text PTT cells
into a clean ordered list of items -- deciding State vs Action, status,
where a "finding" payload belongs -- is the hard, ambiguous, language-
understanding part, and is delegated to an LLM (see llm_ptt_parser.py).
Turning that classified list into the actual graph -- node ids, colors,
per-status shading, edge wiring, statistics -- is a fixed transformation
with one right answer, so it stays plain, deterministic Python: the same
item list always produces the exact same graph, it costs nothing to run,
and it can't "hallucinate" a wrong edge type or color.

Schema is kept 1:1 compatible with the original ptt_parser.build_row_graph()
output (same as processed_graph.zip), so to_html() and everything
downstream keeps working unchanged.
"""

from ptt_parser import (
    STATE_SHADES, ACTION_SHADES, FINDING_COLOR,
    STATE_TRANSITION_COLOR, SEARCH_UPDATE_COLOR, TRACK_UPDATE_COLOR, PREDICTION_COLOR,
    NON_ACTION_LABEL_RE, IP_LABEL_WITH_VALUE_RE, PHASE_NAME_RE, short,
)

VALID_STATUSES = {"completed", "in_progress", "to_do", "unknown"}
VALID_TYPES = {"State", "Action"}


def _norm_item(item):
    """Defensively normalize one item dict (from the LLM parser or its
    regex fallback) before it touches graph assembly. Never raises --
    anything malformed just degrades to a sane default rather than
    crashing a 1900-row batch job on one bad item."""
    number = str(item.get("number") or "").strip() or "0"
    title = (item.get("title") or "").strip() or "Untitled step"

    node_type = item.get("node_type") or item.get("type") or "State"
    if node_type not in VALID_TYPES:
        node_type = "State"

    status = str(item.get("status") or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if status not in VALID_STATUSES:
        status = "unknown"

    payload = item.get("finding")
    if payload is None:
        payload = item.get("payload")
    if isinstance(payload, str):
        payload = payload.strip() or None
    else:
        payload = None

    return {"number": number, "title": title, "type": node_type, "status": status, "payload": payload}


def build_graph_from_items(machine, row_index, items, extra_meta=None):
    """Build one row's attack graph from a classified item list.

    items: list of dicts, each either the LLM schema
        {"number", "title", "node_type", "status", "finding"}
    or the legacy regex-parser schema
        {"number", "title", "type", "status", "payload"}
    (both are accepted; see _norm_item).
    """
    items = [_norm_item(it) for it in items]

    nodes, edges = [], []
    node_ids = set()

    def add_node(id_, label, ntype, color, title, status=None, size=40):
        if id_ in node_ids:
            return
        node_ids.add(id_)
        nodes.append({
            "id": id_, "label": label, "type": ntype, "color": color,
            "status": status, "title": title, "size": size, "shape": "dot",
            "borderColor": "#212529", "borderWidth": 2,
            "font": {"color": "#212529"},
        })

    def add_edge(f, t, label, etype, color, width=2):
        edges.append({
            "from": f, "to": t, "label": label, "type": etype, "color": color,
            "width": width, "arrows": "to", "smooth": {"type": "continuous"},
        })

    start_id = f"state:{machine}:r{row_index}:START"
    add_node(start_id, "State: Start", "State", STATE_SHADES["in_progress"],
             f"Start of pentest tree snapshot for '{machine}' (row {row_index}).",
             status="in_progress", size=30)

    current_state = start_id
    seen_numbers = set()

    for item in items:
        # Guard against the LLM emitting the same numbered item twice.
        dedupe_key = (item["type"], item["number"])
        if dedupe_key in seen_numbers:
            continue
        seen_numbers.add(dedupe_key)

        prefix = "state" if item["type"] == "State" else "action"
        node_id = f"{prefix}:{machine}:r{row_index}:{item['number']}"

        if item["type"] == "State":
            color = STATE_SHADES.get(item["status"], STATE_SHADES["unknown"])
            title = f"{item['number']} {item['title']} [{item['status']}]"
            if item["payload"]:
                title += f"\n{item['payload']}"
            add_node(node_id, f"State {item['number']}\n{short(item['title'], 30)}",
                     "State", color, title, status=item["status"])
            add_edge(current_state, node_id, item["number"], "StateTransition",
                     STATE_TRANSITION_COLOR, 3)
            current_state = node_id
            continue

        # ---- Action node ----
        color = ACTION_SHADES.get(item["status"], ACTION_SHADES["unknown"])
        a_title = f"{item['number']} {item['title']} [{item['status']}]"
        add_node(node_id, f"Action {item['number']}\n{short(item['title'], 30)}",
                 "Action", color, a_title, status=item["status"])
        add_edge(current_state, node_id, f"{item['number']} {short(item['title'], 18)}",
                 "SearchUpdate", SEARCH_UPDATE_COLOR, 2)

        if item["payload"]:
            finding_id = f"finding:{machine}:r{row_index}:{item['number']}"
            add_node(finding_id, f"Finding {item['number']}\n{short(item['payload'], 30)}",
                     "Finding", FINDING_COLOR, item["payload"], status=None, size=32)
            add_edge(node_id, finding_id, "Discover", "TrackUpdate", TRACK_UPDATE_COLOR, 2)
            add_edge(finding_id, current_state, "Leads to", "Prediction", PREDICTION_COLOR, 1)
        else:
            add_edge(node_id, current_state, "Leads to", "Prediction", PREDICTION_COLOR, 1)

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "state_nodes": sum(1 for n in nodes if n["type"] == "State"),
        "action_nodes": sum(1 for n in nodes if n["type"] == "Action"),
        "finding_nodes": sum(1 for n in nodes if n["type"] == "Finding"),
        "ptt_items_parsed": len(items),
    }

    graph = {
        "machine": machine,
        "row_index": row_index,
        "legend": {
            "node_types": {
                "State (Blue)": "Pentest phase / sub-phase, or informational item "
                                 "(target IP, hostname, machine name, OS, ...)",
                "Action (Orange)": "A concrete action the pentester performed",
                "Finding (Green)": "The result/finding produced by an Action",
            },
            "status_shading": {
                "dark": "completed", "mid": "in_progress", "light": "to_do",
            },
            "edge_types": {
                "StateTransition (Black)": "State -> State, advancing through the PTT",
                "SearchUpdate (Green)": "State -> Action, starting work on a PTT item",
                "TrackUpdate (Blue)": "Action -> Finding, item execution produced findings",
                "Prediction (Purple)": "Finding -> State, findings lead back into state",
            },
        },
        "nodes": nodes,
        "edges": edges,
        "graph_statistics": stats,
    }
    if extra_meta:
        graph.update(extra_meta)
    return graph


# ------------------------------------------------------------------------
# Validation -- automated correctness checks, run on every row, every
# pipeline (rule-only, LLM, hybrid). This is what makes "don't ship a
# graph the user has to manually catch problems in" an enforced guarantee
# rather than a hope: any row that fails a check below is collected into
# a report instead of silently passing through. It does not replace the
# classification quality itself (that's the parser/LLM/hybrid's job) --
# it catches assembly/consistency bugs and the one classification error
# the dataset owner called out as a hard rule (identity fields must never
# become Action).
# ------------------------------------------------------------------------
def validate_row_graph(items, graph, machine, row_index):
    """Check structural + hard-rule invariants for one row's graph.

    Returns a list of human-readable problem strings (empty = clean).
    Never raises -- callers decide what to do with a non-empty result
    (log it, fail the build, etc.), so one bad row can't crash a batch.
    """
    problems = []
    items = [_norm_item(it) for it in items]

    # 1. Every source item produced exactly one graph node (no item
    #    silently dropped or duplicated during assembly).
    seen = set()
    dupes = []
    for it in items:
        key = (it["type"], it["number"])
        if key in seen:
            dupes.append(key)
        seen.add(key)
    expected_node_ids = {
        f"{'state' if it['type'] == 'State' else 'action'}:{machine}:r{row_index}:{it['number']}"
        for it in items
    }
    graph_node_ids = {n["id"] for n in graph["nodes"] if not n["id"].endswith(":START")}
    action_or_state_ids = {n["id"] for n in graph["nodes"] if n["type"] in ("State", "Action")}
    missing = expected_node_ids - action_or_state_ids
    if missing:
        problems.append(f"{len(missing)} item(s) missing a corresponding node, e.g. {sorted(missing)[:3]}")
    if dupes:
        problems.append(f"{len(dupes)} duplicate (type, number) item pair(s), e.g. {dupes[:3]}")

    # 2. Every Action item that HAS a payload has exactly one Finding node,
    #    and every Finding node traces back to an Action.
    action_items_with_payload = [it for it in items if it["type"] == "Action" and it["payload"]]
    finding_ids = {n["id"] for n in graph["nodes"] if n["type"] == "Finding"}
    for it in action_items_with_payload:
        fid = f"finding:{machine}:r{row_index}:{it['number']}"
        if fid not in finding_ids:
            problems.append(f"Action {it['number']} ({it['title']!r}) has a payload but no Finding node")
    action_ids = {n["id"] for n in graph["nodes"] if n["type"] == "Action"}
    orphan_findings = [
        fid for fid in finding_ids
        if fid.replace("finding:", "action:", 1) not in action_ids
    ]
    if orphan_findings:
        problems.append(f"{len(orphan_findings)} Finding node(s) with no matching Action, e.g. {orphan_findings[:3]}")

    # 3. HARD RULE: identity/location titles (Target IP, hostname, machine
    #    name, OS, domain, ...) must never be classified Action, no matter
    #    which parser produced the item -- this is the one class of error
    #    explicitly called out as unambiguous, so it's enforced here as a
    #    last line of defense independent of which mode built the graph.
    for it in items:
        title = it["title"].strip()
        if it["type"] == "Action" and (
            NON_ACTION_LABEL_RE.match(title)
            or IP_LABEL_WITH_VALUE_RE.match(title)
            or PHASE_NAME_RE.match(title)
        ):
            problems.append(
                f"HARD RULE VIOLATION: {it['number']} ({it['title']!r}) is an identity/location "
                f"or phase-name field but was classified Action"
            )

    # 4. Status values and node/edge counts are internally consistent with
    #    the graph's own reported statistics (catches assembly bugs that
    #    would otherwise only surface as a visually-wrong HTML).
    stats = graph["graph_statistics"]
    real_counts = {
        "state_nodes": sum(1 for n in graph["nodes"] if n["type"] == "State"),
        "action_nodes": sum(1 for n in graph["nodes"] if n["type"] == "Action"),
        "finding_nodes": sum(1 for n in graph["nodes"] if n["type"] == "Finding"),
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
    }
    for k, v in real_counts.items():
        if stats.get(k) != v:
            problems.append(f"stat mismatch: {k} reported={stats.get(k)} actual={v}")

    return problems
