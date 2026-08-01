#!/usr/bin/env python3
"""
Generate per-row processed attack graphs from data/training_data.csv and data/test_data.csv.
Outputs JSON and HTML files under processed_graphs/train/ and processed_graphs/test/.
"""

import re
import os
import csv
import json
import pathlib
import traceback

# -- Constants --
TRAIN_CSV = "./data/training_data.csv"
TEST_CSV  = "./data/test_data.csv"
OUT_DIR   = "./processed_graphs"

VIS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"

SIZE_STATE   = 40
SIZE_ACTION  = 28
SIZE_FINDING = 20

STATE_COLORS = {
    'completed':   '#1a4a8a',
    'in-progress': '#3A86FF',
    'to-do':       '#93c5fd',
    '':            '#3A86FF',
}
ACTION_COLORS = {
    'completed':   '#7c2d00',
    'in-progress': '#FB5607',
    'to-do':       '#fdb48f',
    '':            '#FB5607',
}
FINDING_COLOR = '#06D6A0'

EDGE_COLORS = {
    'StateTransition': '#111111',
    'SearchUpdate':    '#06D6A0',
    'TrackUpdate':     '#3A86FF',
    'Prediction':      '#8338EC',
}
EDGE_WIDTHS = {
    'StateTransition': 3,
    'SearchUpdate':    2,
    'TrackUpdate':     2,
    'Prediction':      2,
}

_PHASE_KEYWORDS = [
    'reconnaissance', 'recon', 'information gathering',
    'passive information gathering', 'active information gathering',
    'scanning', 'port scanning', 'enumeration', 'web enumeration',
    'service enumeration', 'directory enumeration',
    'exploitation', 'initial access', 'initial foothold', 'foothold',
    'gaining access', 'remote code execution', 'rce',
    'privilege escalation', 'privesc',
    'post-exploitation', 'post exploitation',
    'lateral movement', 'pivoting', 'persistence',
    'exfiltration', 'data exfiltration', 'credential access',
    'credential harvesting', 'covering tracks', 'cleanup',
    'capture the flag', 'capture flag',
    'vulnerability assessment', 'vulnerability scanning',
    'password cracking', 'brute force', 'social engineering', 'phishing',
    'web application', 'sql injection', 'cross-site', 'xss',
    'buffer overflow', 'command injection', 'file inclusion',
    'directory traversal', 'lfi', 'rfi',
]

_IP_RE = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){2,3}\b')
_HOST_KEYWORDS = ['target', 'machine', 'host', 'server', 'victim', 'attacker', 'kali']


def _is_phase(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in _PHASE_KEYWORDS)


def _is_host_or_ip(title: str) -> bool:
    t = title.strip()
    if _IP_RE.search(t):
        return True
    tl = t.lower()
    return any(kw in tl for kw in _HOST_KEYWORDS)


def _classify_item(item) -> str:
    if item['depth'] == 0:
        return 'State'
    if _is_phase(item['title']):
        return 'State'
    if _is_host_or_ip(item['title']):
        return 'State'
    if not item['payload']:
        return 'State'
    return 'Action'


def _safe_machine(val: str):
    """Return cleaned machine name or None if it looks like bad data."""
    if not val or not val.strip():
        return None
    v = val.strip()
    if len(v) > 100:
        return None
    if '\n' in v or '\r' in v:
        return None
    if re.match(r'^\d+\.\s', v):
        return None
    if '{' in v or '}' in v:
        return None
    return v


# -- Status helpers --

_STATUS_EXTRACT_RE = re.compile(
    r'[\(\[]\s*'
    r'(completed|complete|done|to-?do|to\s+do|pending|not\s+started|in[- ]progress|in\s+progress|inprogress)'
    r'\s*[\)\]]',
    re.IGNORECASE
)
_STATUS_BRACE_RE = re.compile(r'\{[Ss]tatus:\s*(?P<s>[^}]+)\}', re.I)

_STATUS_STRIP_RE = re.compile(
    r'\s*[-]?\s*[\(\[]\s*'
    r'(?:completed|complete|done|to-?do|to\s+do|pending|not\s+started|in[- ]progress|in\s+progress|inprogress)'
    r'\s*[\)\]]\s*',
    re.IGNORECASE
)


def _clean_label(title: str) -> str:
    """Strip any status marker from the display label."""
    return _STATUS_STRIP_RE.sub('', title).strip().rstrip('-').strip()


def _normalise_status(raw: str) -> str:
    s = (raw or '').lower().strip()
    s = re.sub(r'\s+', ' ', s)
    if s in ('completed', 'complete', 'done'):
        return 'completed'
    if s in ('in-progress', 'in progress', 'inprogress'):
        return 'in-progress'
    if s in ('to-do', 'to do', 'todo', 'pending', 'not started'):
        return 'to-do'
    return ''


def _node_color(ntype: str, status: str) -> str:
    if ntype == 'State':
        return STATE_COLORS.get(status, STATE_COLORS[''])
    if ntype == 'Action':
        return ACTION_COLORS.get(status, ACTION_COLORS[''])
    return FINDING_COLOR


# CHANGE 1: All node label text is always black
def _font_color(ntype: str, status: str) -> str:
    return '#111111'


def _border_dashes(status: str):
    if status == 'to-do':
        return [5, 5]
    if status == 'in-progress':
        return [2, 2]
    return False


# -- PTT parser --
_HEADER_RE = re.compile(
    r'^(?P<indent>\s*)'
    r'(?P<number>[\d]+(?:\.[\d]+)*\.?)\s+'
    r'(?P<title_raw>.+)$'
)

_INLINE_FINDINGS_RE = re.compile(r'\{(?P<content>[^}]{10,})\}')


def _extract_inline_findings(title_raw: str):
    m = _INLINE_FINDINGS_RE.search(title_raw)
    if not m:
        return title_raw, ''
    content = m.group('content').strip()
    if ':' not in content:
        return title_raw, ''
    clean_title = title_raw[:m.start()].strip().rstrip('-').strip()
    return clean_title, content


def _parse_ptt(ptt_text: str):
    items = []
    current_item = None
    payload_lines = []

    def flush():
        nonlocal current_item, payload_lines
        if current_item is not None:
            extra = '\n'.join(payload_lines).strip()
            if extra:
                if current_item['payload']:
                    current_item['payload'] += '\n' + extra
                else:
                    current_item['payload'] = extra
            items.append(current_item)
        current_item = None
        payload_lines = []

    for raw_line in (ptt_text or '').splitlines():
        line = raw_line.rstrip()
        m = _HEADER_RE.match(line)
        if m:
            flush()
            number    = m.group('number').rstrip('.')
            depth     = number.count('.')
            title_raw = m.group('title_raw').strip()

            # Step 1: Extract status from (parens) or [brackets]
            status_raw = ''
            ms = _STATUS_EXTRACT_RE.search(title_raw)
            if ms:
                status_raw = ms.group(1)
                title_raw  = (title_raw[:ms.start()].strip() + ' ' + title_raw[ms.end():].strip()).strip()

            # Step 2: Extract {Status: ...} brace status
            if not status_raw:
                mb = _STATUS_BRACE_RE.search(title_raw)
                if mb:
                    status_raw = mb.group('s')
                    title_raw  = _STATUS_BRACE_RE.sub('', title_raw, count=1).strip()

            # Step 3: Save raw title for tooltip
            raw_title_for_tooltip = title_raw

            # Step 4: Extract inline findings from {key: val} braces
            clean_title, inline_payload = _extract_inline_findings(title_raw)

            # Step 5: Apply _clean_label to strip any leftover status markers
            display_label = _clean_label(clean_title.strip(' -'))

            current_item = {
                'number':    number,
                'depth':     depth,
                'title':     display_label,
                'raw_title': raw_title_for_tooltip,
                'status':    _normalise_status(status_raw),
                'payload':   inline_payload,
            }
        else:
            if line.strip():
                payload_lines.append(line.strip())

    flush()
    return items


# -- Graph builder --
def _short(text: str, maxlen: int = 60) -> str:
    return text if len(text) <= maxlen else text[:maxlen - 1] + '...'


def build_graph(ptt_text: str, machine: str, row_index: int) -> dict:
    items = _parse_ptt(ptt_text)
    safe  = re.sub(r'[^a-zA-Z0-9_]', '_', machine.lower())
    tag   = f"r{row_index}"

    nodes = []
    edges = []
    node_ids_seen = set()

    def add_node(nid, label, ntype, status, tooltip='', size=None):
        if nid in node_ids_seen:
            return
        node_ids_seen.add(nid)
        color  = _node_color(ntype, status)
        fcol   = _font_color(ntype, status)
        bdash  = _border_dashes(status)
        sz     = size or (SIZE_STATE if ntype == 'State' else
                          SIZE_ACTION if ntype == 'Action' else SIZE_FINDING)
        nodes.append({
            'id':           nid,
            'label':        label,
            'type':         ntype,
            'status':       status,
            'color':        color,
            'size':         sz,
            'title':        tooltip or label,
            'borderWidth':  2,
            'borderDashes': bdash,
            'font':         {'size': 14, 'color': fcol},
        })

    def add_edge(src, tgt, etype):
        edges.append({
            'from':  src,
            'to':    tgt,
            'type':  etype,
            'color': EDGE_COLORS[etype],
            'width': EDGE_WIDTHS[etype],
        })

    if not items:
        nid = f"state:{safe}:{tag}_nodata"
        add_node(nid, '(no PTT data)', 'State', '')
        return {
            'machine':    machine,
            'row_index':  row_index,
            'nodes':      nodes,
            'edges':      edges,
            'stats':      {'total_nodes': 1, 'total_edges': 0,
                           'state_nodes': 1, 'action_nodes': 0, 'finding_nodes': 0},
        }

    classified = [{**it, 'ctype': _classify_item(it)} for it in items]

    state_nodes_in_order = []
    state_action_map = {}
    current_state_nid = None

    for it in classified:
        ctype  = it['ctype']
        status = it['status']
        nid    = f"{ctype.lower()}:{safe}:{tag}_{it['number'].replace('.', '_')}"

        if ctype == 'State':
            # CHANGE 2a: Richer tooltip for State nodes
            tooltip = (
                f"<b>STATE</b> [{it['number']}]<br/>"
                f"<b>{it['title']}</b><br/>"
                f"Status: {status or 'unknown'}"
            )
            if it.get('payload'):
                tooltip += f"<br/><br/><i>Notes:</i><br/>{it['payload'][:500]}"
            label   = _short(it['title'])
            add_node(nid, label, 'State', status, tooltip, SIZE_STATE)
            state_nodes_in_order.append(nid)
            state_action_map[nid] = []
            current_state_nid = nid

        else:  # Action
            # CHANGE 2b: Richer tooltip for Action nodes
            tooltip_action = (
                f"<b>ACTION</b> [{it['number']}]<br/>"
                f"<b>{it['title']}</b><br/>"
                f"Status: {status or 'unknown'}"
            )
            if it['payload']:
                tooltip_action += f"<br/><br/><i>Findings:</i><br/>{it['payload'][:500]}"
            label_action = _short(it['title'])
            add_node(nid, label_action, 'Action', status, tooltip_action, SIZE_ACTION)

            finding_nid = None
            if it['payload']:
                finding_nid = f"finding:{safe}:{tag}_{it['number'].replace('.', '_')}"
                finding_label = _short(it['payload'], 50)
                # CHANGE 2c: Richer tooltip for Finding nodes
                finding_tooltip = (
                    f"<b>FINDING</b> for [{it['number']}]<br/>"
                    f"<b>{it['title']}</b><br/><br/>"
                    f"{it['payload'][:600]}"
                )
                add_node(finding_nid, finding_label, 'Finding', '', finding_tooltip, SIZE_FINDING)
                add_edge(nid, finding_nid, 'TrackUpdate')

            if current_state_nid is not None:
                state_action_map[current_state_nid].append((nid, finding_nid))

    # StateTransition edges: consecutive states
    for i in range(1, len(state_nodes_in_order)):
        add_edge(state_nodes_in_order[i - 1], state_nodes_in_order[i], 'StateTransition')

    # SearchUpdate edges: state -> each action child
    for s_nid, actions in state_action_map.items():
        for (a_nid, f_nid) in actions:
            add_edge(s_nid, a_nid, 'SearchUpdate')

    n_states   = sum(1 for n in nodes if n['type'] == 'State')
    n_actions  = sum(1 for n in nodes if n['type'] == 'Action')
    n_findings = sum(1 for n in nodes if n['type'] == 'Finding')

    return {
        'machine':   machine,
        'row_index': row_index,
        'nodes':     nodes,
        'edges':     edges,
        'stats': {
            'total_nodes':   len(nodes),
            'total_edges':   len(edges),
            'state_nodes':   n_states,
            'action_nodes':  n_actions,
            'finding_nodes': n_findings,
        },
    }


# -- HTML generator --
_LEGEND_HTML = """
    <div id="legend">
      <div class="legend-title">Node Types</div>
      <div class="legend-item"><span class="swatch" style="background:#1a4a8a"></span>State (completed)</div>
      <div class="legend-item"><span class="swatch" style="background:#3A86FF"></span>State (in-progress)</div>
      <div class="legend-item"><span class="swatch" style="background:#93c5fd;border:1px solid #999"></span><span style="color:#333">State (to-do)</span></div>
      <div class="legend-item"><span class="swatch" style="background:#7c2d00"></span>Action (completed)</div>
      <div class="legend-item"><span class="swatch" style="background:#FB5607"></span>Action (in-progress)</div>
      <div class="legend-item"><span class="swatch" style="background:#fdb48f;border:1px solid #999"></span><span style="color:#333">Action (to-do)</span></div>
      <div class="legend-item"><span class="swatch" style="background:#06D6A0"></span>Finding</div>
    </div>
"""

# CHANGE 3: Added .vis-tooltip CSS; CHANGE 4: hover/tooltipDelay updated in interaction
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{machine} -- Row {row_index}</title>
  <script src="{vis_cdn}"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f8f9fa; }}
    #network {{ width: 100%; height: 80vh; background: #fff; border-bottom: 1px solid #ddd; }}
    #legend {{ position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.95);
               border: 1px solid #ccc; border-radius: 6px; padding: 10px 14px; font-size: 13px; z-index: 99; }}
    .legend-title {{ font-weight: bold; margin-bottom: 6px; }}
    .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
    .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 50%;
               margin-right: 7px; flex-shrink: 0; border: 1px solid transparent; }}
    #info {{ padding: 8px 14px; font-size: 12px; color: #555; }}
    div.vis-tooltip {{
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #cccccc !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        font-family: 'Segoe UI', Arial, sans-serif !important;
        font-size: 13px !important;
        max-width: 380px !important;
        white-space: pre-wrap !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
        pointer-events: none !important;
    }}
  </style>
</head>
<body>
  <div id="network"></div>
  {legend}
  <div id="info">
    <strong>{machine}</strong> | Row {row_index} |
    Nodes: {total_nodes} (States: {state_nodes}, Actions: {action_nodes}, Findings: {finding_nodes}) |
    Edges: {total_edges}
  </div>
  <script>
    var nodes = new vis.DataSet({nodes_json});
    var edges = new vis.DataSet({edges_json});
    var container = document.getElementById("network");
    var options = {{
      nodes: {{ shape: "dot", borderWidth: 2 }},
      edges: {{ arrows: "to", smooth: {{ type: "dynamic" }} }},
      physics: {{ stabilization: {{ iterations: 200 }} }},
      interaction: {{ hover: true, tooltipDelay: 150, navigationButtons: true, keyboard: true }},
    }};
    new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
  </script>
</body>
</html>"""


def build_html(graph: dict) -> str:
    stats = graph['stats']
    nodes_vis = []
    for n in graph['nodes']:
        nodes_vis.append({
            'id':    n['id'],
            'label': n['label'],
            'color': {'background': n['color'], 'border': '#333',
                      'highlight': {'background': n['color'], 'border': '#000'}},
            'size':  n['size'],
            'title': n.get('title', n['label']),
            'font':  n.get('font', {'size': 14, 'color': '#111111'}),
            'borderWidth':  n.get('borderWidth', 2),
            'borderDashes': n.get('borderDashes', False),
        })
    edges_vis = []
    for e in graph['edges']:
        edges_vis.append({
            'from':  e['from'],
            'to':    e['to'],
            'color': {'color': e['color']},
            'width': e['width'],
            'title': e.get('type', ''),
        })
    return _HTML_TEMPLATE.format(
        machine      = graph['machine'],
        row_index    = graph['row_index'],
        vis_cdn      = VIS_CDN,
        legend       = _LEGEND_HTML,
        total_nodes  = stats['total_nodes'],
        state_nodes  = stats['state_nodes'],
        action_nodes = stats['action_nodes'],
        finding_nodes= stats['finding_nodes'],
        total_edges  = stats['total_edges'],
        nodes_json   = json.dumps(nodes_vis),
        edges_json   = json.dumps(edges_vis),
    )


# -- Main --
def process_split(csv_path: str, split_name: str, out_root: str):
    out_dir = pathlib.Path(out_root) / split_name
    machine_counters = {}
    rows_done = 0
    machines_seen = set()
    errors = 0
    skipped = 0

    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    print(f"Processing {split_name} split: {csv_path} ...")

    for row in rows:
        raw_machine = row.get('Machine') or ''
        machine = _safe_machine(raw_machine)
        if not machine:
            skipped += 1
            continue
        ptt_text = row.get('PTT') or ''

        machine_counters.setdefault(machine, 0)
        machine_counters[machine] += 1
        row_index = machine_counters[machine]

        mdir = out_dir / machine
        mdir.mkdir(parents=True, exist_ok=True)

        try:
            graph = build_graph(ptt_text, machine, row_index)
            json_path = mdir / f"row_{row_index}_graph.json"
            html_path = mdir / f"row_{row_index}_graph.html"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, indent=2)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(build_html(graph))
            rows_done += 1
            machines_seen.add(machine)
        except Exception as e:
            errors += 1
            print(f"  ERROR row {row_index} machine={machine}: {e}")
            traceback.print_exc()

        if rows_done % 200 == 0:
            print(f"  [{split_name}] processed {rows_done}/{len(rows)} rows ...")

    print(f"  Done: {rows_done} rows, {len(machines_seen)} machines"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {errors} errors" if errors else ""))


def main():
    print("=== generate_processed_graphs.py ===\n")
    process_split(TRAIN_CSV, 'train', OUT_DIR)
    process_split(TEST_CSV,  'test',  OUT_DIR)
    print("\nAll done.")


if __name__ == '__main__':
    main()
