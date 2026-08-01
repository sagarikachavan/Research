#!/usr/bin/env python3
"""
Build input/train.json and input/test.json from CSV data.
For each row in data/training_data.csv and data/test_data.csv,
produces a JSON record with the same graph structure as generate_processed_graphs.py.
"""

import re
import csv
import json
import pathlib

# -- Constants --
TRAIN_CSV = "./data/training_data.csv"
TEST_CSV  = "./data/test_data.csv"
OUT_DIR   = "./input"

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
    'reconnaissance', 'recon',
    'information gathering', 'passive information gathering', 'active information gathering',
    'port scanning',
    'web enumeration', 'service enumeration', 'directory enumeration',
    'exploitation', 'initial access', 'initial foothold', 'foothold', 'gaining access',
    'privilege escalation', 'privesc',
    'post-exploitation', 'post exploitation',
    'lateral movement', 'pivoting',
    'persistence',
    'exfiltration', 'data exfiltration',
    'credential access', 'credential harvesting',
    'covering tracks', 'cleanup',
    'vulnerability assessment',
    'capture the flag', 'capture flag',
]

_IP_RE = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){2,3}\b')
_HOST_KEYWORDS = ['target', 'machine', 'host', 'server', 'victim', 'attacker', 'kali']


def _is_phase(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in _PHASE_KEYWORDS)


def _is_host_or_ip(title: str) -> bool:
    if _IP_RE.search(title):
        return True
    tl = title.lower()
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


def _font_color(ntype: str, status: str) -> str:
    if ntype == 'Finding':
        return '#1e1e1e'
    if status == 'to-do':
        return '#1e1e1e'
    return '#ffffff'


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
        color = _node_color(ntype, status)
        fcol  = _font_color(ntype, status)
        bdash = _border_dashes(status)
        sz    = size or (SIZE_STATE if ntype == 'State' else
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
            'machine':   machine,
            'row_index': row_index,
            'nodes':     nodes,
            'edges':     edges,
            'stats':     {'total_nodes': 1, 'total_edges': 0,
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
            tooltip = f"[{it['number']}] {it.get('raw_title', it['title'])} [{status or 'unknown'}]"
            add_node(nid, _short(it['title']), 'State', status, tooltip, SIZE_STATE)
            state_nodes_in_order.append(nid)
            state_action_map[nid] = []
            current_state_nid = nid
        else:
            tooltip_action = f"[{it['number']}] {it.get('raw_title', it['title'])}\nStatus: {status or 'unknown'}"
            if it['payload']:
                tooltip_action += f"\n\nFindings:\n{it['payload'][:400]}"
            add_node(nid, _short(it['title']), 'Action', status, tooltip_action, SIZE_ACTION)

            finding_nid = None
            if it['payload']:
                finding_nid = f"finding:{safe}:{tag}_{it['number'].replace('.', '_')}"
                add_node(finding_nid, _short(it['payload'], 50), 'Finding', '',
                         it['payload'][:600], SIZE_FINDING)
                add_edge(nid, finding_nid, 'TrackUpdate')

            if current_state_nid is not None:
                state_action_map[current_state_nid].append((nid, finding_nid))

    for i in range(1, len(state_nodes_in_order)):
        add_edge(state_nodes_in_order[i - 1], state_nodes_in_order[i], 'StateTransition')

    for s_nid, actions in state_action_map.items():
        for a_nid, _ in actions:
            add_edge(s_nid, a_nid, 'SearchUpdate')

    for i, s_nid in enumerate(state_nodes_in_order[:-1]):
        next_s  = state_nodes_in_order[i + 1]
        actions = state_action_map.get(s_nid, [])
        if actions:
            last_a, last_f = actions[-1]
            src = last_f if last_f else last_a
        else:
            src = s_nid
        add_edge(src, next_s, 'Prediction')

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


def _safe_machine(val: str):
    if not val or not val.strip():
        return None
    v = val.strip()
    if len(v) > 60:
        return None
    if re.match(r'^\d+\.\s', v):
        return None
    return v


def process_csv(csv_path: str, out_path: pathlib.Path):
    records = []
    machine_row_count = {}
    total_action_nodes = 0

    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            machine = _safe_machine(row.get('Machine', ''))
            if not machine:
                continue

            machine_row_count.setdefault(machine, 0)
            machine_row_count[machine] += 1
            row_index = machine_row_count[machine]

            ptt_text = row.get('PTT') or ''
            graph    = build_graph(ptt_text, machine, row_index)
            total_action_nodes += graph['stats']['action_nodes']

            records.append({
                'machine':      machine,
                'graph':        graph,
                'new_strategy': row.get('New strategy', '').strip(),
                'row_index':    row_index,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(records, fh, indent=2)

    n_machines = len(machine_row_count)
    print(f"  {csv_path} -> {out_path}: {len(records)} records from {n_machines} machines, {total_action_nodes} action-nodes total")
    return len(records), n_machines, total_action_nodes


def main():
    out = pathlib.Path(OUT_DIR)
    train_recs, train_m, train_a = process_csv(TRAIN_CSV, out / 'train.json')
    test_recs,  test_m,  test_a  = process_csv(TEST_CSV,  out / 'test.json')
    print("Done.")
    print(f"\nSummary:")
    print(f"  Train: {train_recs} records, {train_m} machines, {train_a} action-nodes")
    print(f"  Test:  {test_recs} records, {test_m} machines, {test_a} action-nodes")


if __name__ == '__main__':
    main()
