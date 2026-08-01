"""
ptt_graph.py
============
Core logic to turn a single PTT (Penetration Testing Tree) cell from the
dataset into an attack-graph representation with three node types:

    STATE   (blue)   - a pentest phase / sub-phase, or purely informational
                        items (target IP, machine name, hostname, OS, etc.)
                        that are not an action the operator performed.
    ACTION  (orange) - a concrete action the pentester performed
                        (port scan, enumeration, exploitation, ...).
    FINDING (green)  - the result/finding produced by an ACTION.

Edges (matching the scheme requested):
    StateTransition (black)  : State  -> State   (advancing through the PTT)
    SearchUpdate    (green)  : State  -> Action   (starting work on a PTT item)
    TrackUpdate     (blue)   : Action -> Finding  (execution produced findings)
    Prediction      (purple) : Finding-> State    (findings lead back into the
                                                    current pentest state)

Node "status" (completed / in_progress / to_do) is preserved on STATE and
ACTION nodes and drives a dark / mid / light shade of the base color so the
three statuses are visually distinguishable.
"""

import re

# ----------------------------------------------------------------------
# Machine-name validity check
# ----------------------------------------------------------------------
# A handful of rows in the source CSVs have their columns shifted -- the
# PTT tree text leaks into the "Machine" column (e.g. an unescaped
# character upstream threw off the CSV parser), so "Machine" ends up
# holding something like "1. Reconnaissance {completed}\n1.1 Port
# scanning ..." instead of an actual machine name. These rows have no
# usable machine name and their other fields are misaligned too, so they
# should be excluded rather than turned into garbage output folders.
MACHINE_FRAGMENT_RE = re.compile(
    r"^\s*\d+[\.\)]|\(to-?do\)|\[to-?do\]|\(completed\)|\[completed\]|"
    r"\(in[- ]progress\)|\[in[- ]progress\]|\{(to-?do|completed|in[- ]progress|"
    r"status|findings)\b",
    re.IGNORECASE,
)
MAX_MACHINE_NAME_LEN = 40


def is_valid_machine_name(name) -> bool:
    """True if `name` looks like an actual machine name rather than a
    PTT-tree fragment that leaked into the Machine column."""
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name or name.lower() == "nan":
        return False
    if len(name) > MAX_MACHINE_NAME_LEN:
        return False
    if "\n" in name:
        return False
    if MACHINE_FRAGMENT_RE.search(name):
        return False
    return True


# ----------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------
STATE_BASE = "#3A86FF"     # blue
ACTION_BASE = "#FB5607"    # orange
FINDING_COLOR = "#06D6A0"  # green

STATE_TRANSITION_COLOR = "#000000"   # black
SEARCH_UPDATE_COLOR = "#06D6A0"      # green
TRACK_UPDATE_COLOR = "#3A86FF"       # blue
PREDICTION_COLOR = "#8338EC"         # purple

# Shades: completed = dark, in_progress = mid (base color), to_do = light
STATE_SHADES = {
    "completed": "#0B3D91",
    "in_progress": "#3A86FF",
    "to_do": "#AFC8FF",
    "unknown": "#3A86FF",
}
ACTION_SHADES = {
    "completed": "#B33E00",
    "in_progress": "#FB5607",
    "to_do": "#FFC7A3",
    "unknown": "#FB5607",
}

# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
HEADER_RE = re.compile(
    r'^[ \t]*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*(?:[-\u2013]\s*)?'
    r'(?:[\(\[](completed|to-do|to do|in[- ]progress)[\)\]]'
    r'|\{\s*Status:\s*(completed|to-do|to do|in[- ]progress)\s*\})'
    r'(.*)$',
    re.IGNORECASE | re.MULTILINE,
)

# Word / stem fragments that indicate the PTT item describes an ACTION the
# pentester actually performed, as opposed to a purely informational /
# state-like item (target IP, hostname, OS, machine name, etc.)
ACTION_STEMS = [
    "scan", "enumerat", "explor", "exploit", "check", "determin", "obtain",
    "updat", "perform", "identif", "gain", "escalat", "download", "upload",
    "crack", "brute", "decod", "decrypt", "inject", "bypass", "extract",
    "transfer", "execut", "run", "test", "verify", "confirm", "analyz",
    "review", "search", "discover", "access", "connect", "attack",
    "harvest", "dump", "list", "view", "read", "writ", "modif", "creat",
    "install", "configur", "examin", "investigat", "assess", "look",
    "find", "locat", "retriev", "captur", "monitor", "quer", "prob",
    "fuzz", "spray", "guess", "attempt", "establish", "generat",
    "request", "send", "submit", "login", "log in", "authenticat",
    "navigat", "browse", "interact", "clone", "compile", "reverse",
    "pivot", "tunnel", "spoof", "sniff", "intercept", "manipulat",
]
ACTION_STEM_RE = re.compile("|".join(ACTION_STEMS), re.IGNORECASE)


def _norm_status(raw):
    if not raw:
        return "unknown"
    raw = raw.lower().replace("-", " ").strip()
    if "complet" in raw:
        return "completed"
    if "progress" in raw:
        return "in_progress"
    if "to do" in raw or raw == "todo":
        return "to_do"
    return "unknown"


def _clean(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_ptt(text):
    """Parse a PTT cell into an ordered list of item dicts (document order)."""
    text = text if isinstance(text, str) else ""
    items = []
    matches = list(HEADER_RE.finditer(text))
    for idx, m in enumerate(matches):
        num, title, status_a, status_b, tail = m.groups()
        status = _norm_status(status_a or status_b)
        start_extra = m.end()
        end_extra = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        extra = text[start_extra:end_extra]
        payload = (tail + extra).strip()
        payload = re.sub(r"^[:\s]+", "", payload)
        payload = re.sub(r"^\{?\s*Findings?:\s*", "", payload, flags=re.IGNORECASE)
        payload = payload.strip().lstrip("{").rstrip("}").strip()

        title_clean = _clean(title)
        # Some PTT rows omit a human title and go straight to a `{...}` info
        # block (e.g. "1.3.2.1 {Target IP: ..., Findings: ...} - (completed)").
        # In that case the whole block was greedily captured as the "title" --
        # pull it out into the payload and derive a short label instead.
        if title_clean.startswith("{"):
            info = title_clean.strip("{}").strip()
            payload = (info + (" " + payload if payload else "")).strip()
            label_match = re.match(r"^([^,:]{1,40}):", info)
            title_clean = label_match.group(1).strip() if label_match else "Details"

        items.append({
            "number": num,
            "title": title_clean,
            "status": status,
            "payload": payload if payload else None,
            "depth": num.count("."),
        })
    return items


def classify(item):
    """Return 'state', 'action' for a parsed PTT item (finding is separate)."""
    if item["depth"] == 0:
        return "state"
    if not item["payload"]:
        return "state"
    if ACTION_STEM_RE.search(item["title"]):
        return "action"
    return "state"


def short(text, n=70):
    text = _clean(text)
    return text if len(text) <= n else text[: n - 1] + "\u2026"


# ----------------------------------------------------------------------
# Graph building (one graph per PTT cell / CSV row)
# ----------------------------------------------------------------------
def build_row_graph(machine, row_index, ptt_text, mcp_tasks_raw=None, extra_meta=None):
    items = parse_ptt(ptt_text)

    nodes = []
    edges = []
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

    for item in items:
        node_type = classify(item)
        node_id = f"{node_type}:{machine}:r{row_index}:{item['number']}"

        if node_type == "state":
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

        # ---- action node ----
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
            add_edge(node_id, finding_id, "Discover", "TrackUpdate",
                      TRACK_UPDATE_COLOR, 2)
            add_edge(finding_id, current_state, "Leads to", "Prediction",
                      PREDICTION_COLOR, 1)
        else:
            add_edge(node_id, current_state, "Leads to", "Prediction",
                      PREDICTION_COLOR, 1)

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


# ----------------------------------------------------------------------
# HTML visualization (vis-network), reusing the style of the original script
# ----------------------------------------------------------------------
def to_html(graph):
    import json as _json

    machine = graph["machine"]
    row_index = graph["row_index"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    stats = graph["graph_statistics"]

    vis_nodes = _json.dumps([
        {"id": n["id"], "label": n["label"], "color": n["color"], "shape": "dot",
         "size": n["size"], "borderColor": n["borderColor"], "borderWidth": n["borderWidth"],
         "font": n["font"], "title": n["title"]}
        for n in nodes
    ])
    vis_edges = _json.dumps([
        {"from": e["from"], "to": e["to"], "label": e["label"], "color": e["color"],
         "width": e["width"], "arrows": e["arrows"], "smooth": e["smooth"], "title": e["type"]}
        for e in edges
    ])

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{machine} - row {row_index} - PTT Attack Graph</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f8f9fa; }}
    header {{ padding: 14px 18px; background: white; border-bottom: 1px solid #d1d5db; }}
    #network {{ height: calc(100vh - 100px); width: 100vw; background-color: #f8f9fa; border: 1px solid lightgray; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px; margin-top: 6px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  </style>
</head>
<body>
  <header>
    <strong>{machine} &mdash; row {row_index} &mdash; PTT Attack Graph</strong>
    ({stats['state_nodes']} state, {stats['action_nodes']} action, {stats['finding_nodes']} finding nodes)
    <div class="legend">
      <span><span class="swatch" style="background:#0B3D91"></span>State (dark=completed)</span>
      <span><span class="swatch" style="background:#3A86FF"></span>State (mid=in progress)</span>
      <span><span class="swatch" style="background:#AFC8FF"></span>State (light=to-do)</span>
      <span><span class="swatch" style="background:#B33E00"></span>Action (dark=completed)</span>
      <span><span class="swatch" style="background:#FB5607"></span>Action (mid=in progress)</span>
      <span><span class="swatch" style="background:#FFC7A3"></span>Action (light=to-do)</span>
      <span><span class="swatch" style="background:#06D6A0"></span>Finding</span>
      <span>Black: StateTransition | Green: SearchUpdate | Blue: TrackUpdate | Purple: Prediction</span>
    </div>
  </header>
  <div id="network"></div>
  <script type="text/javascript">
    var nodes = new vis.DataSet({vis_nodes});
    var edges = new vis.DataSet({vis_edges});
    var container = document.getElementById('network');
    var data = {{ nodes: nodes, edges: edges }};
    var options = {{
      physics: {{ forceAtlas2Based: {{ springLength: 220, springConstant: 0.03, centralGravity: 0.008, damping: 0.35, nodeDistance: 180 }},
                  minVelocity: 0.75, solver: "forceAtlas2Based", stabilization: {{ iterations: 220 }} }},
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
      nodes: {{ font: {{ size: 12 }} }},
      edges: {{ font: {{ size: 9, color: '#212529', align: 'middle' }}, smooth: {{ type: 'continuous' }} }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""