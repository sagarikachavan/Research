#!/usr/bin/env python3
"""
Show training data samples from graph structure training.
"""
import json
import os
import random
from typing import List, Dict

from config import ROOT, RANDOM_SEED
from data_utils import build_graph_from_input_json_graph

random.seed(RANDOM_SEED)


def load_graphs_from_directory(graph_dir: str) -> List[dict]:
    """Load all graph JSON files from a directory."""
    graph_dicts = []
    
    for machine_dir in os.listdir(graph_dir):
        machine_path = os.path.join(graph_dir, machine_dir)
        if not os.path.isdir(machine_path):
            continue
        
        for fname in os.listdir(machine_path):
            if fname.endswith("_graph.json"):
                graph_path = os.path.join(machine_path, fname)
                with open(graph_path, 'r') as f:
                    graph_dicts.append(json.load(f))
    
    return graph_dicts


def build_adjacency_dict(graph_dict: dict) -> dict:
    """Build adjacency dictionary from graph edges."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    
    node_id_to_label = {n["id"]: n.get("label", n.get("title", "")) for n in nodes}
    
    adjacency = {n["id"]: [] for n in nodes}
    for edge in edges:
        src = edge["from"]
        tgt = edge["to"]
        if src in adjacency:
            adjacency[src].append(tgt)
        if tgt in adjacency:
            adjacency[tgt].append(src)
    
    return adjacency, node_id_to_label


def show_adjacency_samples(graph_dict: dict, num_samples: int = 3):
    """Show adjacency prediction training samples."""
    print("\n" + "=" * 80)
    print("ADJACENCY PREDICTION TASK SAMPLES")
    print("=" * 80)
    
    nodes = graph_dict.get("nodes", [])
    adjacency, node_id_to_label = build_adjacency_dict(graph_dict)
    
    count = 0
    for node in nodes:
        if count >= num_samples:
            break
        
        node_id = node["id"]
        adj_nodes = adjacency.get(node_id, [])
        
        if not adj_nodes:
            continue
        
        num_targets = min(len(adj_nodes), random.randint(1, 3))
        target_nodes = random.sample(adj_nodes, num_targets)
        
        prompt = f"""Given the following graph node:
Node ID: {node_id}
Node Label: {node_id_to_label[node_id]}

Based on the graph structure encoded in the soft prompt tokens, predict which nodes are directly connected (adjacent) to this node. List the adjacent node IDs.

Adjacent nodes:"""
        
        target = ", ".join(target_nodes)
        
        print(f"\n--- Sample {count + 1} ---")
        print(f"INPUT (Prompt):")
        print(prompt)
        print(f"\nGROUND TRUTH (Target):")
        print(target)
        print(f"\nGround truth adjacent nodes: {adj_nodes}")
        print(f"Selected for training: {target_nodes}")
        
        count += 1


def show_node_type_samples(graph_dict: dict, num_samples: int = 3):
    """Show node type prediction training samples."""
    print("\n" + "=" * 80)
    print("NODE TYPE PREDICTION TASK SAMPLES")
    print("=" * 80)
    
    nodes = graph_dict.get("nodes", [])
    node_id_to_label = {n["id"]: n.get("label", n.get("title", "")) for n in nodes}
    
    count = 0
    for node in nodes:
        if count >= num_samples:
            break
        
        node_id = node["id"]
        node_type = node.get("type", "Agent")
        
        prompt = f"""Given the following graph node:
Node ID: {node_id}
Node Label: {node_id_to_label[node_id]}

Based on the graph structure encoded in the soft prompt tokens, predict the node type (Agent, Search, or Track).

Node type:"""
        
        target = node_type
        
        print(f"\n--- Sample {count + 1} ---")
        print(f"INPUT (Prompt):")
        print(prompt)
        print(f"\nGROUND TRUTH (Target):")
        print(target)
        
        count += 1


def show_edge_type_samples(graph_dict: dict, num_samples: int = 3):
    """Show edge type prediction training samples."""
    print("\n" + "=" * 80)
    print("EDGE TYPE PREDICTION TASK SAMPLES")
    print("=" * 80)
    
    edges = graph_dict.get("edges", [])
    node_id_to_label = {n["id"]: n.get("label", n.get("title", "")) for n in graph_dict.get("nodes", [])}
    
    count = 0
    for edge in edges:
        if count >= num_samples:
            break
        
        src_id = edge["from"]
        tgt_id = edge["to"]
        edge_type = edge.get("type", "StateTransition")
        
        prompt = f"""Given the following graph edge:
From Node: {src_id} ({node_id_to_label.get(src_id, "Unknown")})
To Node: {tgt_id} ({node_id_to_label.get(tgt_id, "Unknown")})

Based on the graph structure encoded in the soft prompt tokens, predict the edge type.

Edge type:"""
        
        target = edge_type
        
        print(f"\n--- Sample {count + 1} ---")
        print(f"INPUT (Prompt):")
        print(prompt)
        print(f"\nGROUND TRUTH (Target):")
        print(target)
        
        count += 1


def show_path_samples(graph_dict: dict, num_samples: int = 3):
    """Show path prediction training samples."""
    print("\n" + "=" * 80)
    print("PATH PREDICTION TASK SAMPLES (2-hop)")
    print("=" * 80)
    
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    adjacency, node_id_to_label = build_adjacency_dict(graph_dict)
    
    count = 0
    for node in nodes:
        if count >= num_samples:
            break
        
        node_id = node["id"]
        adj_nodes = adjacency.get(node_id, [])
        
        if not adj_nodes:
            continue
        
        for adj_id in adj_nodes:
            adj_adj_nodes = adjacency.get(adj_id, [])
            if not adj_adj_nodes:
                continue
            
            target_id = random.choice(adj_adj_nodes)
            if target_id == node_id:
                continue
            
            prompt = f"""Given the following graph node:
Node ID: {node_id}
Node Label: {node_id_to_label[node_id]}

Based on the graph structure encoded in the soft prompt tokens, predict a node that is 2 hops away (connected through one intermediate node).

2-hop node ID:"""
            
            target = target_id
            
            print(f"\n--- Sample {count + 1} ---")
            print(f"INPUT (Prompt):")
            print(prompt)
            print(f"\nGROUND TRUTH (Target):")
            print(target)
            print(f"\nPath: {node_id} -> {adj_id} -> {target_id}")
            
            count += 1
            break


def main():
    print("=" * 80)
    print("TRAINING DATA SAMPLES FOR GRAPH STRUCTURE TASKS")
    print("=" * 80)
    
    # Load graphs
    graph_dir = os.path.join(ROOT, "processed_data", "train")
    print(f"\nLoading graphs from: {graph_dir}")
    graph_dicts = load_graphs_from_directory(graph_dir)
    print(f"Loaded {len(graph_dicts)} graphs")
    
    # Use first graph for samples
    if graph_dicts:
        graph_dict = graph_dicts[0]
        print(f"\nUsing graph: {graph_dict.get('nodes', [{}])[0].get('label', 'Unknown') if graph_dict.get('nodes') else 'Unknown'}")
        print(f"  Nodes: {len(graph_dict.get('nodes', []))}")
        print(f"  Edges: {len(graph_dict.get('edges', []))}")
        
        # Show samples for each task
        show_adjacency_samples(graph_dict, num_samples=2)
        show_node_type_samples(graph_dict, num_samples=2)
        show_edge_type_samples(graph_dict, num_samples=2)
        show_path_samples(graph_dict, num_samples=2)
    else:
        print("No graphs found!")
    
    print("\n" + "=" * 80)
    print("SAMPLE DISPLAY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
