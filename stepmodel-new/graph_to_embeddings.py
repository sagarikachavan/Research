#!/usr/bin/env python3
"""
Convert graph JSON files to node/edge embeddings and structured dataset JSON.
"""
import os
import json
import re
from typing import Dict, List, Any, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


def parse_ptt(ptt_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse PTT text into a dictionary of items."""
    pattern = re.compile(r'^[ \t]*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*(?:[-\u2013]\s*)?(?:[\(\[](completed|to-do|to do|in[- ]progress)[\)\]]|\{\s*Status:\s*(completed|to-do|to do|in[- ]progress)\s*\})(.*)$', re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(ptt_text))
    items = {}
    for idx, m in enumerate(matches):
        num, title, status_a, status_b, tail = m.groups()
        status = (status_a or status_b or "").lower()
        start_extra = m.end()
        end_extra = matches[idx+1].start() if idx+1 < len(matches) else len(ptt_text)
        extra = ptt_text[start_extra:end_extra]
        payload = (tail + extra).strip()
        payload = re.sub(r'^[:\s]+', '', payload)
        payload = re.sub(r'^\{?\s*Findings:\s*', '', payload, flags=re.IGNORECASE)
        payload = payload.strip().lstrip("{").rstrip("}").strip()
        items[num] = {
            "number": num,
            "title": title.strip(),
            "status": status,
            "payload": payload if payload else None,
            "depth": num.count(".")
        }
    return items


def load_graph_json(json_path: str) -> Dict[str, Any]:
    """Load graph data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_node_text(node: Dict[str, Any]) -> str:
    """Combine node fields into a single text string for embedding."""
    return f"{node.get('type', '')} {node.get('label', '')} {node.get('title', '')}".strip()


def get_edge_text(edge: Dict[str, Any], nodes: Dict[str, Dict[str, Any]]) -> str:
    """Combine edge fields with node info into a single text string for embedding."""
    src_node = nodes.get(edge.get('from'), {})
    tgt_node = nodes.get(edge.get('to'), {})
    return f"{edge.get('type', '')} {edge.get('label', '')} {src_node.get('title', '')} {tgt_node.get('title', '')}".strip()


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Embed a list of texts using the sentence transformer model."""
    if not texts:
        return np.array([])
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


BAD_MACHINE_PATTERN = re.compile(
    r"\n|Status:|Findings:|^\s*\d+(?:\.\d+)*\.?\s+|\(completed\)|\[completed\]|\(to-?do\)|\[to-?do\]",
    re.IGNORECASE,
)

STEP_ID_PATTERN = re.compile(r":r(?P<run>\d+)_s(?P<step>\d+)_")


def load_clean_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a CSV and explicitly drop rows whose Machine field contains PTT text."""
    df = pd.read_csv(csv_path)
    machine_values = df["Machine"].fillna("").astype(str)
    bad_mask = machine_values.str.len().gt(80) | machine_values.str.contains(BAD_MACHINE_PATTERN, regex=True)
    dropped = df.loc[bad_mask].copy()
    clean = df.loc[~bad_mask].copy()
    report = {
        "csv_path": csv_path,
        "rows_total": int(len(df)),
        "rows_dropped_bad_machine": int(len(dropped)),
        "dropped_row_indices": [int(i) for i in dropped.index.tolist()],
    }
    if report["rows_dropped_bad_machine"]:
        print(
            f"Warning: dropped {report['rows_dropped_bad_machine']} rows from "
            f"{os.path.basename(csv_path)} with corrupted Machine fields."
        )
    return clean, report


def _step_index_from_graph_id(raw_id: str):
    match = STEP_ID_PATTERN.search(str(raw_id))
    if not match:
        return 0
    return int(match.group("step"))


def _filter_graph_to_step(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], cutoff_step: int):
    """Return the cumulative graph visible through cutoff_step."""
    kept_nodes = [
        node for node in nodes
        if _step_index_from_graph_id(node.get("id", "")) <= cutoff_step
    ]
    kept_ids = {node.get("id") for node in kept_nodes}
    kept_edges = [
        edge for edge in edges
        if edge.get("from") in kept_ids
        and edge.get("to") in kept_ids
        and max(
            _step_index_from_graph_id(edge.get("from", "")),
            _step_index_from_graph_id(edge.get("to", "")),
        ) <= cutoff_step
    ]
    return kept_nodes, kept_edges


def _embed_graph(nodes_list, edges_list, embed_model: SentenceTransformer):
    node_dict = {n['id']: n for n in nodes_list}
    node_texts = [get_node_text(n) for n in nodes_list]
    node_embeddings = embed_texts(node_texts, embed_model)
    node_data = []
    for i, n in enumerate(nodes_list):
        node_data.append({
            "id": n['id'],
            "label": n.get('label', ''),
            "type": n.get('type', ''),
            "title": n.get('title', ''),
            "embedding": node_embeddings[i].tolist() if len(node_embeddings) > 0 else []
        })

    edge_texts = [get_edge_text(e, node_dict) for e in edges_list]
    edge_embeddings = embed_texts(edge_texts, embed_model)
    edge_data = []
    for i, e in enumerate(edges_list):
        edge_data.append({
            "from": e.get('from', ''),
            "to": e.get('to', ''),
            "label": e.get('label', ''),
            "type": e.get('type', ''),
            "embedding": edge_embeddings[i].tolist() if len(edge_embeddings) > 0 else []
        })
    return node_data, edge_data


def process_machine_graph(graph_data: Dict[str, Any], df: pd.DataFrame, machine: str, embed_model: SentenceTransformer) -> Dict[str, Any]:
    """Process a single machine's graph into embeddings and structured data."""
    machine_df = df[df['Machine'] == machine].sort_index()
    if machine_df.empty:
        machine_df = df[df['Machine'].astype(str).str.replace('_', ' ', regex=False) == machine.replace('_', ' ')].sort_index()
    step_pairs = []
    nodes_list = graph_data.get('nodes', [])
    edges_list = graph_data.get('edges', [])
    for i in range(len(machine_df) - 1):
        prev_row = machine_df.iloc[i]
        next_row = machine_df.iloc[i + 1]
        # cutoff_step=i means pair 0 sees only START/baseline; pair 1 sees graph growth from row 0, etc.
        step_nodes, step_edges = _filter_graph_to_step(nodes_list, edges_list, cutoff_step=i)
        embedded_nodes, embedded_edges = _embed_graph(step_nodes, step_edges, embed_model)
        step_pairs.append({
            "nodes": embedded_nodes,
            "edges": embedded_edges,
            "graph_cutoff_step": i,
            "previous_strategy": str(prev_row.get('New strategy', '')),
            "previous_strategy_explanation": str(prev_row.get('Strategy explanation', '')),
            "previous_step": str(prev_row.get('New step', '')),
            "previous_step_explanation": str(prev_row.get('Step explanation', '')),
            "previous_step_result": str(prev_row.get('Previous step result', '')),
            "previous_mcp_tasks": str(prev_row.get('MCP_tasks', '')),
            "next_strategy": str(next_row.get('New strategy', '')),
            "next_strategy_explanation": str(next_row.get('Strategy explanation', '')),
            "next_step": str(next_row.get('New step', '')),
            "next_step_explanation": str(next_row.get('Step explanation', '')),
            "next_mcp_tasks": str(next_row.get('MCP_tasks', ''))
        })

    node_data, edge_data = _embed_graph(nodes_list, edges_list, embed_model)

    return {
        "machine": machine,
        "graph_statistics": graph_data.get('graph_statistics', {}),
        "nodes": node_data,
        "edges": edge_data,
        "step_pairs": step_pairs
    }


def process_directory(input_dir: str, csv_path: str, output_dir: str, embed_model: SentenceTransformer):
    """Process all machine graphs in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    df, cleanup_report = load_clean_csv(csv_path)

    for machine_dir_name in os.listdir(input_dir):
        machine_dir = os.path.join(input_dir, machine_dir_name)
        if not os.path.isdir(machine_dir):
            continue

        # Find graph JSON file
        json_files = [f for f in os.listdir(machine_dir) if f.endswith('_graph.json')]
        if not json_files:
            print(f"Skipping {machine_dir_name}: no graph JSON found")
            continue

        json_path = os.path.join(machine_dir, json_files[0])
        print(f"Processing {machine_dir_name}...")

        try:
            # Get machine name from filename (remove _graph.json)
            machine_name = json_files[0].replace('_graph.json', '')
            # Fix underscores
            machine_name = machine_name.replace('_', ' ')

            graph_data = load_graph_json(json_path)
            processed = process_machine_graph(graph_data, df, machine_name, embed_model)
            processed["csv_cleanup"] = cleanup_report
            all_data.append(processed)

            # Save individual machine data
            machine_out_path = os.path.join(output_dir, f"{machine_dir_name}_processed.json")
            with open(machine_out_path, 'w', encoding='utf-8') as f:
                json.dump(processed, f, indent=2)

        except Exception as e:
            print(f"Error processing {machine_dir_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save all data in one file
    all_out_path = os.path.join(output_dir, "all_processed.json")
    with open(all_out_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved all processed data to {all_out_path}")
    return all_data


def assert_no_machine_leakage(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]):
    train_machines = {str(item.get("machine", "")).strip() for item in train_data if item.get("machine")}
    test_machines = {str(item.get("machine", "")).strip() for item in test_data if item.get("machine")}
    overlap = train_machines & test_machines
    assert overlap == set(), f"Machine leakage between train/test after embedding regeneration: {sorted(overlap)}"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    processed_graphs_dir = os.path.join(base_dir, "processed_data")
    embeddings_dir = os.path.join(base_dir, "embeddings_data")

    # Load embedding model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process train data
    print("\n=== Processing train data ===")
    train_input = os.path.join(processed_graphs_dir, "train")
    train_csv = os.path.join(data_dir, "training_data.csv")
    train_output = os.path.join(embeddings_dir, "train")
    if os.path.exists(train_input):
        train_data = process_directory(train_input, train_csv, train_output, model)
    else:
        train_data = []

    # Process test data
    print("\n=== Processing test data ===")
    test_input = os.path.join(processed_graphs_dir, "test")
    test_csv = os.path.join(data_dir, "test_data.csv")
    test_output = os.path.join(embeddings_dir, "test")
    if os.path.exists(test_input):
        test_data = process_directory(test_input, test_csv, test_output, model)
    else:
        test_data = []

    assert_no_machine_leakage(train_data, test_data)

    print("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()
