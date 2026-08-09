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
    STATE_TRANSITION_COLOR, ACTION_UPDATE_COLOR, FINDING_UPDATE_COLOR, PREDICTION_COLOR,
    short,
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
                 "ActionUpdate", ACTION_UPDATE_COLOR, 2)

        if item["payload"]:
            finding_id = f"finding:{machine}:r{row_index}:{item['number']}"
            add_node(finding_id, f"Finding {item['number']}\n{short(item['payload'], 30)}",
                     "Finding", FINDING_COLOR, item["payload"], status=None, size=32)
            add_edge(node_id, finding_id, "Discover", "FindingUpdate", FINDING_UPDATE_COLOR, 2)
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
                "ActionUpdate (Green)": "State -> Action, starting work on a PTT item",
                "FindingUpdate (Blue)": "Action -> Finding, item execution produced findings",
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
