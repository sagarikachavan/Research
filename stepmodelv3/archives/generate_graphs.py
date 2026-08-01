#!/usr/bin/env python3
"""
Generate graph datasets from data/training_data.csv and data/test_data.csv.
For each machine, creates a directory with graph JSON and HTML visualization
following the same structure as stepmodel/graph_dataset/pentest-dataset.
"""
import json
import re
import os
import ast
import pandas as pd

AGENT = "#3A86FF"
GOAL = "#FF006E"
SEARCH = "#FB5607"
TRACK = "#06D6A0"
BLACK = "#000000"
GREEN = "#06D6A0"
BLUE = "#3A86FF"
PURPLE = "#8338EC"

FRAGMENT_PATTERN = re.compile(
    r"^\d+[\.\)]|\(to-?do\)|\[to-?do\]|\(completed\)|\[completed\]|\(in progress\)|\[in progress\]",
    re.I,
)

HEADER_RE = re.compile(
    r'^[ \t]*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*(?:[-\u2013]\s*)?'
    r'(?:[\(\[](completed|to-do|to do|in[- ]progress)[\)\]]'
    r'|\{\s*Status:\s*(completed|to-do|to do|in[- ]progress)\s*\})'
    r'(.*)$',
    re.IGNORECASE | re.MULTILINE,
)

END_PATTERN = re.compile(r"end (the )?(pentest|task)|generate the report", re.I)


def short(text, n=60):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def safe(text):
    return "" if not isinstance(text, str) else text


def parse_ptt(text):
    text = safe(text)
    matches = list(HEADER_RE.finditer(text))
    items = {}
    for idx, m in enumerate(matches):
        num, title, status_a, status_b, tail = m.groups()
        status = (status_a or status_b or "").lower()
        start_extra = m.end()
        end_extra = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        extra = text[start_extra:end_extra]
        payload = (tail + extra).strip()
        payload = re.sub(r"^[:\s]+", "", payload)
        payload = re.sub(r"^\{?\s*Findings:\s*", "", payload, flags=re.IGNORECASE)
        payload = payload.strip().lstrip("{").rstrip("}").strip()
        items[num] = {
            "number": num,
            "title": title.strip(),
            "status": status,
            "payload": payload if payload else None,
            "depth": num.count("."),
        }
    return items


def phase_title(number, items_lookup):
    parts = number.split(".")
    if len(parts) >= 2:
        parent_num = ".".join(parts[:2])
    else:
        parent_num = parts[0]
    parent = items_lookup.get(parent_num)
    if parent and parent_num != number:
        return parent["title"]
    top = items_lookup.get(parts[0])
    return top["title"] if top else ""


def format_tree(items):
    lines = []
    for num in sorted(items.keys(), key=lambda s: [int(p) for p in s.split(".")]):
        it = items[num]
        indent = "  " * it["depth"]
        mark = "✓" if "complet" in it["status"] else ("…" if "progress" in it["status"] else "○")
        lines.append(f"{indent}{mark} {num} {it['title']}")
    return "\n".join(lines)


def format_mcp(mcp_raw):
    try:
        d = ast.literal_eval(safe(mcp_raw))
        if isinstance(d, dict):
            return "\n".join(f"- {k}: {v}" for k, v in d.items())
    except Exception:
        pass
    return safe(mcp_raw)


def load_valid_machines(csv_path):
    df = pd.read_csv(csv_path)
    vc = df["Machine"].value_counts()
    valid_names = [m for m in vc.index if len(m) < 40 and not FRAGMENT_PATTERN.search(m)]
    return df, valid_names


def detect_runs(sub):
    runs = []
    current = [0]
    for i in range(1, len(sub)):
        prev_strat = safe(sub.loc[i, "Previous strategy"]).strip()
        prior_new_strat = safe(sub.loc[i - 1, "New strategy"]).strip()
        if prev_strat != prior_new_strat:
            runs.append(current)
            current = [i]
        else:
            current.append(i)
    runs.append(current)
    return runs


def leaf_new_items(prev_items, next_items):
    new_nums = [n for n in next_items if n not in prev_items]
    leafy = [next_items[n] for n in new_nums if next_items[n]["payload"]]
    if leafy:
        return sorted(leafy, key=lambda it: [int(p) for p in it["number"].split(".")])
    return sorted([next_items[n] for n in new_nums],
                  key=lambda it: [int(p) for p in it["number"].split(".")])


def build_machine_graph(df, machine):
    sub = df[df["Machine"] == machine].reset_index(drop=True)
    runs = detect_runs(sub)

    nodes, edges = [], []
    node_ids = set()

    def add_node(id_, label, ntype, color, title, size=40):
        if id_ in node_ids:
            return
        node_ids.add(id_)
        nodes.append({
            "id": id_, "label": label, "type": ntype, "color": color,
            "title": title, "size": size, "shape": "dot",
            "borderColor": "#212529", "borderWidth": 3, "font": {"color": "#212529"}
        })

    def add_edge(f, t, label, etype, color, width=3):
        edges.append({
            "from": f, "to": t, "label": label, "type": etype, "color": color,
            "width": width, "arrows": "to", "smooth": {"type": "continuous"}
        })

    start_id = f"agent:{machine}:START"
    add_node(start_id, "Agent: Initial\n(Start)", "Agent", AGENT,
              f"Start of pentest against '{machine}'. {len(runs)} recorded run(s) below.")

    for r_idx, run_rows in enumerate(runs, start=1):
        run_tag = f"[Run {r_idx}/{len(runs)}] " if len(runs) > 1 else ""
        row_objs = [sub.loc[idx] for idx in run_rows]
        item_snapshots = [parse_ptt(row["PTT"]) for row in row_objs]
        k = len(row_objs)

        prev_agent = start_id

        baseline_items = {n: it for n, it in item_snapshots[0].items() if it["payload"]}
        if baseline_items:
            base_agent_id = f"agent:{machine}:r{r_idx}_base"
            base_search_id = f"search:{machine}:r{r_idx}_base"
            base_track_id = f"track:{machine}:r{r_idx}_base"
            first_items = sorted(baseline_items.values(),
                                  key=lambda it: [int(p) for p in it["number"].split(".")])
            add_node(base_search_id, f"Search: Baseline\n({len(first_items)} item(s))", "Search", SEARCH,
                      f"{run_tag}Baseline recon already completed before this run's first tracked step:\n" +
                      "\n".join(f"{it['number']} {it['title']}" for it in first_items))
            add_node(base_track_id, f"Track: {short(first_items[-1]['title'], 26)}", "Track", TRACK,
                      f"{run_tag}" + "\n\n".join(f"{it['number']} {it['title']}: {it['payload']}" for it in first_items),
                      size=34)
            add_node(base_agent_id, f"Agent: Baseline\n{short(phase_title(first_items[-1]['number'], item_snapshots[0]), 22)}",
                      "Agent", AGENT, f"{run_tag}Cumulative PTT so far:\n{format_tree(item_snapshots[0])}")
            add_edge(prev_agent, base_agent_id, "baseline", "StateTransition", BLACK, 4)
            add_edge(prev_agent, base_search_id, "Baseline Recon", "SearchUpdate", GREEN)
            add_edge(base_search_id, base_track_id, "Discover", "TrackUpdate", BLUE)
            add_edge(base_track_id, base_agent_id, "Leads to", "Prediction", PURPLE, 2)
            prev_agent = base_agent_id

        for i in range(k):
            row = row_objs[i]
            current_items = item_snapshots[i]
            is_last = i == k - 1
            new_items = [] if is_last else leaf_new_items(current_items, item_snapshots[i + 1])
            cumulative = item_snapshots[i + 1] if not is_last else current_items

            if not new_items:
                goal = is_last and bool(END_PATTERN.search(safe(row.get("New strategy")) + safe(row.get("New step"))))
                if goal or is_last:
                    close_id = f"agent:{machine}:r{r_idx}_close"
                    add_node(close_id, "Agent: Task\nComplete (Goal)" if goal else "Agent: Run End\n(no further PTT growth)",
                              "Agent", GOAL if goal else AGENT,
                              f"{run_tag}Final cumulative PTT for this run:\n{format_tree(cumulative)}", size=46)
                    add_edge(prev_agent, close_id, "close", "StateTransition", BLACK, 4)
                continue

            for item in new_items:
                agent_id = f"agent:{machine}:r{r_idx}_s{i+1}_{item['number']}"
                search_id = f"search:{machine}:r{r_idx}_s{i+1}_{item['number']}"
                track_id = f"track:{machine}:r{r_idx}_s{i+1}_{item['number']}"

                mcp_text = format_mcp(row.get("MCP_tasks"))
                add_node(search_id, f"Search: {item['number']}\n{short(item['title'], 26)}", "Search", SEARCH,
                          f"{run_tag}PTT item {item['number']}: {item['title']}\n\nMCP tool calls:\n{mcp_text}")

                payload_snippet = short(item["payload"], 30) if item["payload"] else short(item["title"], 30)
                add_node(track_id, f"Track: {payload_snippet}", "Track", TRACK,
                          f"{run_tag}{item['number']} {item['title']}:\n{item['payload'] or '(status change only)'}",
                          size=34)

                is_goal = is_last is False and END_PATTERN.search(item["title"])
                phase = phase_title(item["number"], cumulative)
                add_node(agent_id, f"Agent: {item['number']}\n{short(phase, 22)}", "Agent",
                          GOAL if is_goal else AGENT,
                          f"{run_tag}Cumulative PTT after {item['number']}:\n{format_tree(cumulative)}",
                          size=46 if is_goal else 40)

                add_edge(prev_agent, agent_id, f"→ {item['number']}", "StateTransition", BLACK, 4)
                add_edge(prev_agent, search_id, f"{item['number']} {short(item['title'], 20)}", "SearchUpdate", GREEN)
                add_edge(search_id, track_id, "Discover", "TrackUpdate", BLUE)
                add_edge(track_id, agent_id, "Leads to", "Prediction", PURPLE, 2)

                prev_agent = agent_id

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "agent_nodes": sum(1 for n in nodes if n["type"] == "Agent"),
        "search_nodes": sum(1 for n in nodes if n["type"] == "Search"),
        "track_nodes": sum(1 for n in nodes if n["type"] == "Track"),
        "runs_detected": len(runs),
        "rows_captured": len(sub),
    }
    return nodes, edges, stats


def to_dict_json(machine, nodes, edges, stats, csv_name):
    return {
        "paper_analogy": "Deep Reinforcement Learning on Graphs for Autonomous Search-and-Track",
        "machine": machine,
        "source": f"{csv_name}, Machine == '{machine}', {stats['rows_captured']} row(s), "
                   f"{stats['runs_detected']} auto-detected run(s)/playthroughs. "
                   f"Graph built directly from the PTT (Penetration Testing Tree) field's growth.",
        "analogy_mapping": {
            "Agent Nodes": "PTT snapshot reached after each new tree item (state = cumulative tree)",
            "Search Nodes": "The PTT tree item worked on (number + title) + its MCP_tasks tool calls",
            "Track Nodes": "That item's findings payload -- what was actually discovered"
        },
        "legend": {
            "node_types": {
                "Agent (Blue/Pink goal)": "PTT state after a new item completes",
                "Search (Orange)": "PTT item being worked (tool calls in tooltip)",
                "Track (Green)": "Findings payload of that item"
            },
            "edge_types": {
                "StateTransition (Black)": "Agent -> Agent, label = PTT item number",
                "SearchUpdate (Green)": "Agent -> Search, item being worked from that state",
                "TrackUpdate (Blue)": "Search -> Track, what item's execution found",
                "Prediction (Purple)": "Track -> Agent, leads to next cumulative PTT state"
            }
        },
        "nodes": nodes,
        "edges": edges,
        "graph_statistics": stats
    }


def to_html(machine, nodes, edges, stats):
    vis_nodes = json.dumps([
        {"id": nd["id"], "label": nd["label"], "color": nd["color"], "shape": "dot",
         "size": nd["size"], "borderColor": nd["borderColor"], "borderWidth": nd["borderWidth"],
         "font": nd["font"], "title": nd["title"]}
        for nd in nodes
    ])
    vis_edges = json.dumps([
        {"from": e["from"], "to": e["to"], "label": e["label"], "color": e["color"],
         "width": e["width"], "arrows": e["arrows"], "smooth": e["smooth"], "title": e["type"]}
        for e in edges
    ])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{machine} - PTT Evolution Graph</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f8f9fa; }}
    header {{ padding: 14px 18px; background: white; border-bottom: 1px solid #d1d5db; }}
    #network {{ height: calc(100vh - 78px); width: 100vw; background-color: #f8f9fa; border: 1px solid lightgray; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; margin-top: 6px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  </style>
</head>
<body>
  <header>
    <strong>{machine} - PTT Evolution Graph ({stats['runs_detected']} run(s), {stats['rows_captured']} row(s))</strong>
    <div class="legend">
      <span><span class="swatch" style="background:#3A86FF"></span>Agent (cumulative PTT state)</span>
      <span><span class="swatch" style="background:#FF006E"></span>Agent Goal</span>
      <span><span class="swatch" style="background:#FB5607"></span>Search (PTT item worked)</span>
      <span><span class="swatch" style="background:#06D6A0"></span>Track (findings)</span>
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
      physics: {{ forceAtlas2Based: {{ springLength: 340, springConstant: 0.02, centralGravity: 0.006, damping: 0.3, nodeDistance: 300 }},
                  minVelocity: 0.75, solver: "forceAtlas2Based", stabilization: {{ iterations: 220 }} }},
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
      nodes: {{ font: {{ size: 13 }} }},
      edges: {{ font: {{ size: 9, color: '#212529', align: 'middle' }}, smooth: {{ type: 'continuous' }} }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""


def sanitize_dirname(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def process_csv(csv_path, output_dir, csv_name):
    df, valid_machines = load_valid_machines(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    skipped = []
    for m in valid_machines:
        nodes, edges, stats = build_machine_graph(df, m)
        mdir = os.path.join(output_dir, sanitize_dirname(m))
        os.makedirs(mdir, exist_ok=True)
        dict_json = to_dict_json(m, nodes, edges, stats, csv_name)
        with open(os.path.join(mdir, f"{sanitize_dirname(m)}_graph.json"), "w") as f:
            json.dump(dict_json, f, indent=2)
        with open(os.path.join(mdir, f"{sanitize_dirname(m)}_graph.html"), "w") as f:
            f.write(to_html(m, nodes, edges, stats))
        summary[m] = stats
    return summary, skipped


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(base_dir, "processed_data")

    print("=== Processing training_data.csv ===")
    train_csv = os.path.join(data_dir, "training_data.csv")
    train_out = os.path.join(processed_dir, "train")
    train_summary, train_skipped = process_csv(train_csv, train_out, "training_data.csv")
    print(f"Built {len(train_summary)} machine graphs in {train_out}")

    print("\n=== Processing test_data.csv ===")
    test_csv = os.path.join(data_dir, "test_data.csv")
    test_out = os.path.join(processed_dir, "test")
    test_summary, test_skipped = process_csv(test_csv, test_out, "test_data.csv")
    print(f"Built {len(test_summary)} machine graphs in {test_out}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
