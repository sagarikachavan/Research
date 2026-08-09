"""
ptt_parser.py
=============
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

-----------------------------------------------------------------------
Why this file replaces the previous parser
-----------------------------------------------------------------------
The previous version matched PTT items with a single regex that REQUIRED an
explicit status token, e.g. "(completed)" or "[to-do]", to sit immediately
after the item title. In practice, a lot of PTT cells don't follow that
exact shape:

  * Some sub-items have no status token of their own at all, e.g.:
        1.3.1 Perform a port scan - {Findings: ...}
        1.3.2 Determine the services and versions on each open port - {Findings: ...}
        1.4 Target IP - 10.129.232.106
    None of these three lines matched the old regex, so they were silently
    swallowed into the *previous* matched item's trailing text -- which is
    exactly the "succession" bug that was reported: 1.3.1's and 1.3.2's
    findings, AND 1.4's "Target IP" text, all ended up jammed into a single
    oversized "Finding 1.3" node instead of being their own nodes.

  * Some items put the status token *after* the {...} block instead of
    before it, e.g.:
        1.3.1 Perform a port scan - {Findings: Open Ports: 22, 80, 443, 8888} - [completed]
    which the old title-capture group would swallow whole (including the
    "{...}" block) because "." doesn't match "{" specially -- so the
    Findings text ended up inside the *title*, not the payload.

  * Some Findings blocks span multiple physical lines with plain "- ..."
    bullets and no numbering of their own, and the closing "}" + status can
    appear several lines below the opening "{".

  * Some items are "bare" data blocks with no title at all, immediately
    nested one level under an action-titled parent that itself has no
    inline payload, e.g.:
        1.3.2 Determine the services and versions on each open port - (completed)
            1.3.2.1 {Target IP: ..., Findings: ...} - (completed)
    Here 1.3.2.1 isn't really its own PTT step -- it *is* the finding that
    belongs to 1.3.2 ("Determine the services...").

This rewrite parses PTT text line-by-line (numbered items always start a
new line in this dataset -- verified against both CSVs), extracts each
item's payload with a brace-balance scan (so multi-line / oddly-placed
status tokens don't break anything), and merges "bare" findings-only child
items back into their action parent so they don't show up as orphan nodes.
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
ACTION_UPDATE_COLOR = "#28a745"      # green
FINDING_UPDATE_COLOR = "#007bff"     # blue
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

# Matches the START of a numbered PTT item. In this dataset every numbered
# item begins its own line (verified empirically), so we split the whole
# cell on these line-starts rather than trying to match a whole item (title
# + status + payload) with one regex.
ITEM_START_RE = re.compile(
    r"^[ \t]*(\d+(?:\.\d+)*)\.?[ \t]+(?=\S)",
    re.MULTILINE,
)

STATUS_RE = re.compile(
    r"[\(\[]\s*(completed|to[- ]do|in[- ]progress)\s*[\)\]]"
    r"|\{\s*Status:\s*(completed|to[- ]do|in[- ]progress)\s*\}",
    re.IGNORECASE,
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

# Short informational field labels that happen to contain an action-stem
# substring (e.g. "Scan Duration" contains "scan") but are really just a
# data field, the same as "Target IP" or "Host Status" -- not something the
# pentester "did". Matched as a whole title (ignoring a trailing
# parenthetical like "(Port 443)") so it doesn't over-match real actions.
NON_ACTION_LABEL_RE = re.compile(
    r"^(scan duration|target ip|ip address|host status|mac address|"
    r"ssh hostkey|ssh version|http server headers?|http titles?|"
    r"ssl certificate|smb version|os version|service version)s?"
    r"(\s*\(.*\))?$",
    re.IGNORECASE,
)


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


def _strip_title_edges(text):
    """Trim trailing separators ('-', ':', whitespace) commonly left over
    once the status token and payload have been removed from a title."""
    text = text.strip()
    text = re.sub(r"[\s\-\u2013:]+$", "", text)
    return text.strip()


def _find_balanced_brace(text, start):
    """Given `text` and the index of an opening '{', return the index of
    its matching '}' using simple brace-depth counting. If the text is
    truncated / malformed and no matching close is found, return the
    length of the text (i.e. "consume to the end")."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _extract_status_outside_payload(pre, post):
    m = STATUS_RE.search(pre)
    if m:
        return _norm_status(m.group(1) or m.group(2))
    m = STATUS_RE.search(post)
    if m:
        return _norm_status(m.group(1) or m.group(2))
    return "unknown"


def _clean_payload(payload):
    payload = payload.strip()
    # Drop a leading "Findings:" / "Findings" label -- the label itself
    # doesn't add information once the text is attached to a Finding node.
    payload = re.sub(r"^\{?\s*Findings?\s*:?\s*", "", payload, flags=re.IGNORECASE)
    payload = payload.strip().rstrip("}").strip()
    return _clean(payload)


def _split_raw_items(text):
    """Split PTT text into raw (number, block_text) pairs using line-start
    numbering as the item boundary. Any preamble before the first numbered
    item (e.g. "Based on the provided results, ...") is discarded."""
    matches = list(ITEM_START_RE.finditer(text))
    raw_items = []
    for idx, m in enumerate(matches):
        num = m.group(1)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw_items.append((num, text[start:end]))
    return raw_items


def _parse_item_block(num, block):
    """Pull title / status / payload out of one item's raw block text."""
    brace_idx = block.find("{")
    if brace_idx == -1:
        status = _extract_status_outside_payload(block, "")
        title = _strip_title_edges(STATUS_RE.sub("", block))
        payload = None
    else:
        close_idx = _find_balanced_brace(block, brace_idx)
        pre = block[:brace_idx]
        inner = block[brace_idx + 1:close_idx]
        post = block[close_idx + 1:] if close_idx < len(block) else ""

        status = _extract_status_outside_payload(pre, post)
        title = _strip_title_edges(STATUS_RE.sub("", pre))
        if status == "unknown":
            # Some malformed/truncated rows only carry the status word
            # inside the payload itself; fall back to searching there too.
            m = STATUS_RE.search(inner)
            if m:
                status = _norm_status(m.group(1) or m.group(2))
        payload = _clean_payload(inner)
        if not payload:
            payload = None

    return {
        "number": num,
        "title": title,
        "status": status,
        "payload": payload,
        "depth": num.count("."),
        "bare": (title == ""),
    }


def parse_ptt(text):
    """Parse a PTT cell into an ordered list of item dicts (document order).

    Handles two dataset quirks beyond plain item extraction:
      * items whose title text is empty (a bare "{...}" data block) are
        merged into the immediately preceding item when that preceding
        item has no payload of its own and the bare item is one level
        deeper -- this is the "1.3.2 -> 1.3.2.1 {..}" pattern where the
        bare child *is* the parent's finding, not a separate PTT step.
      * a bare item that can't be merged (no suitable parent) is kept as
        its own item, with a short label derived from its first "Key:"
        pair so it still renders sensibly.
    """
    text = text if isinstance(text, str) else ""
    raw_items = _split_raw_items(text)
    parsed = [_parse_item_block(num, block) for num, block in raw_items]

    items = []
    for item in parsed:
        if item["bare"] and item["payload"]:
            if (
                items
                and items[-1]["payload"] is None
                and item["depth"] == items[-1]["depth"] + 1
            ):
                items[-1]["payload"] = item["payload"]
                continue
            # No suitable parent to merge into -- keep it, with a label
            # derived from the first "Key:" pair in its payload/title.
            label_match = re.match(r"^([^,:{}]{1,40}):", item["payload"] or "")
            item["title"] = label_match.group(1).strip() if label_match else "Details"
        items.append(item)

    return items


def classify(item):
    """Return 'state' or 'action' for a parsed PTT item ('finding' nodes
    are derived separately, from an action's payload)."""
    if item["depth"] == 0:
        return "state"
    if not item["payload"]:
        return "state"
    if NON_ACTION_LABEL_RE.match(item["title"].strip()):
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
                  "ActionUpdate", ACTION_UPDATE_COLOR, 2)

        if item["payload"]:
            finding_id = f"finding:{machine}:r{row_index}:{item['number']}"
            add_node(finding_id, f"Finding {item['number']}\n{short(item['payload'], 30)}",
                      "Finding", FINDING_COLOR, item["payload"], status=None, size=32)
            add_edge(node_id, finding_id, "Discover", "FindingUpdate",
                      FINDING_UPDATE_COLOR, 2)
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
         "width": e["width"], "arrows": e["arrows"], "smooth": e["smooth"], "title": f"{e['type']}: {e.get('label', '')}"}
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
