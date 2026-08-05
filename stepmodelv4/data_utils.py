"""
data_utils.py — Data loading utilities for stepmodelv4
Combines stepmodelv3 graph format with stepmodelv2-style processing
"""
from __future__ import annotations

import json
import ast
import re
from typing import Optional, List, Dict
import torch
import numpy as np
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

from config import TEXT_EMB_DIM

# Canonical step categories (from stepmodelv3)
STEP_LABELS = [
    "Do a google search for more information",
    "Enumerate further on the X service to find software versions, hidden directories and file.",
    "Explore the suspicious files, commands and create a summary of the findings.",
    "Further Enumerate the website. - hidden directories, links and software",
    "Enumerate the domain",
    "Exploit the selected exploitations",
    "Analyze the outcomes of the previous step and find an attack path",
    "Ask for human assistant",
    "Explore the source code for vulnerabilities.",
    "End task and ask permission to generate the report",
]

_STEP_KEYWORDS = [
    ("Do a google search for more information", ["google search", "search", "research", "cve", "vulnerability"]),
    ("Enumerate further on the X service to find software versions, hidden directories and file.", ["enumerate further", "enumerate", "enumeration", "service", "software version", "hidden directories"]),
    ("Explore the suspicious files, commands and create a summary of the findings.", ["explore the suspicious", "explore", "suspicious files", "commands", "summary", "findings"]),
    ("Further Enumerate the website. - hidden directories, links and software", ["enumerate the website", "website", "web", "hidden directories", "links", "software"]),
    ("Enumerate the domain", ["enumerate the domain", "domain", "dns", "subdomain"]),
    ("Exploit the selected exploitations", ["exploit", "exploitation", "attack", "payload", "shell"]),
    ("Analyze the outcomes of the previous step and find an attack path", ["analyze the outcomes", "analyze outcomes", "attack path", "analyze", "outcome", "result"]),
    ("Ask for human assistant", ["ask for human", "human assistant", "help", "assistant"]),
    ("Explore the source code for vulnerabilities.", ["source code", "code review", "code", "vulnerabilities"]),
    ("End task and ask permission to generate the report", ["end task", "end", "complete", "finish", "report", "document"]),
]

# Canonical MCP tool vocabulary (from stepmodelv3)
MCP_LABELS = [
    "Nmap",
    "Metasploit",
    "Netcat",
    "Dirbuster",
    "SQLmap",
    "Smb client",
    "hydra",
    "John-the-ripper",
    "Google search",
    "Interactive CLI",
    "Web page interaction",
]

_MCP_KEYWORDS = [
    ("Nmap", ["nmap", "netdiscover"]),
    ("Metasploit", ["metasploit", "msf"]),
    ("Netcat", ["netcat", "nc"]),
    ("Dirbuster", ["dirbuster", "gobuster"]),
    ("SQLmap", ["sqlmap"]),
    ("Smb client", ["smb client", "smbclient"]),
    ("hydra", ["hydra"]),
    ("John-the-ripper", ["john", "john the ripper"]),
    ("Google search", ["google", "search"]),
    ("Interactive CLI", ["cli", "interactive", "terminal"]),
    ("Web page interaction", ["web", "browser", "http", "https"]),
]

# Context fields for text embedding (adapted from stepmodelv2)
CONTEXT_COLUMNS = ["machine", "candidate_step", "candidate_step_explanation"]

_sentence_encoder: Optional[SentenceTransformer] = None


def get_sentence_encoder():
    global _sentence_encoder
    if _sentence_encoder is None:
        _sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sentence_encoder


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts using sentence transformer."""
    encoder = get_sentence_encoder()
    embeddings = encoder.encode(texts, convert_to_numpy=True)
    return embeddings  # (n_texts, TEXT_EMB_DIM)


def classify_step(text: str) -> str:
    """Map a free-text step string to one of the 10 canonical STEP_LABELS."""
    if not text or not text.strip():
        return "unknown"
    t = text.lower()
    for canon, kws in _STEP_KEYWORDS:
        if any(kw in t for kw in kws):
            return canon
    return "unknown"


def normalize_tool(tool_name: str) -> Optional[str]:
    """Normalize a tool name to canonical form."""
    if not tool_name:
        return None
    t = tool_name.lower()
    for canon, kws in _MCP_KEYWORDS:
        if any(kw in t for kw in kws):
            return canon
    return None


def mcp_labels_from_dict(mcp_dict: Dict[str, str]) -> List[str]:
    """Extract normalized MCP tool labels from a dictionary."""
    labels = []
    for tool_name in mcp_dict.keys():
        normalized = normalize_tool(tool_name)
        if normalized:
            labels.append(normalized)
    return labels


def mcp_multihot(mcp_labels: List[str]) -> np.ndarray:
    """Convert MCP labels to multi-hot vector."""
    vec = np.zeros(len(MCP_LABELS), dtype=np.float32)
    for label in mcp_labels:
        if label in MCP_LABELS:
            vec[MCP_LABELS.index(label)] = 1.0
    return vec


def parse_completion(completion: str) -> Optional[Dict]:
    """Parse completion string into JSON dictionary."""
    if not completion or not completion.strip():
        return None
    
    # Try to extract JSON from completion
    completion = completion.strip()
    
    # Remove markdown code blocks if present
    if completion.startswith("```"):
        completion = completion[3:]
    if completion.startswith("```json"):
        completion = completion[7:]
    if completion.endswith("```"):
        completion = completion[:-3]
    completion = completion.strip()
    
    # Try parsing as JSON
    try:
        return json.loads(completion)
    except json.JSONDecodeError:
        # Try to find JSON object in the string
        start_idx = completion.find("{")
        end_idx = completion.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(completion[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        return None


def graph_to_torch_geometric(graph_json: Dict) -> Data:
    """Convert stepmodelv3 graph JSON to torch_geometric Data object."""
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])
    
    # Create node features
    node_texts = [node.get("label", "") for node in nodes]
    node_embs = _embed_texts(node_texts)  # (n_nodes, TEXT_EMB_DIM)
    
    # Add node type one-hot encoding (Agent/Search/Track)
    node_types = []
    for node in nodes:
        label = node.get("label", "").lower()
        if "agent" in label:
            node_types.append([1, 0, 0])
        elif "search" in label:
            node_types.append([0, 1, 0])
        elif "track" in label:
            node_types.append([0, 0, 1])
        else:
            node_types.append([0, 0, 0])
    
    node_types = np.array(node_types, dtype=np.float32)
    x = np.concatenate([node_embs, node_types], axis=1)  # (n_nodes, TEXT_EMB_DIM + 3)
    
    # Create edge index
    edge_index = []
    for edge in edges:
        source_idx = None
        target_idx = None
        for i, node in enumerate(nodes):
            if node.get("id") == edge.get("source"):
                source_idx = i
            if node.get("id") == edge.get("target"):
                target_idx = i
        if source_idx is not None and target_idx is not None:
            edge_index.append([source_idx, target_idx])
            edge_index.append([target_idx, source_idx])  # Undirected
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        num_nodes=len(nodes)
    )


def load_json_data(json_path: str) -> List[Dict]:
    """Load JSON data from stepmodelv3 format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # Parse graph JSON if it's a string
        graph_json = item.get("graph", {})
        if isinstance(graph_json, str):
            try:
                graph_json = json.loads(graph_json)
            except:
                graph_json = {}
        
        # Parse MCP tasks if it's a string
        mcp_tasks = item.get("gold_mcp_tasks", {})
        if isinstance(mcp_tasks, str):
            try:
                # Try parsing as Python dict representation
                mcp_tasks = ast.literal_eval(mcp_tasks)
            except:
                # Try parsing as JSON
                try:
                    mcp_tasks = json.loads(mcp_tasks)
                except:
                    mcp_tasks = {}
        
        # Classify step
        step_text = item.get("gold_new_step", "")
        step_category = classify_step(step_text)
        
        # Get MCP labels
        mcp_labels = mcp_labels_from_dict(mcp_tasks)
        mcp_vec = mcp_multihot(mcp_labels)
        
        # Build context fields
        context = {
            "machine": item.get("machine", ""),
            "candidate_step": item.get("new_strategy", ""),
            "candidate_step_explanation": item.get("strategy_explanation", ""),
        }
        
        # Embed context fields
        context_texts = [context.get(c, "") for c in CONTEXT_COLUMNS]
        field_embs = _embed_texts(context_texts)  # (3, TEXT_EMB_DIM)
        
        example = {
            "graph_json": graph_json,
            "graph": graph_to_torch_geometric(graph_json),
            "context": context,
            "field_embs": field_embs,
            "gold_step_text": step_text,
            "gold_step_category": step_category,
            "gold_step_idx": STEP_LABELS.index(step_category) if step_category in STEP_LABELS else 0,
            "gold_mcp_dict": mcp_tasks,
            "gold_mcp_labels": mcp_labels,
            "gold_mcp_vec": mcp_vec,
            "gold_step_explanation": item.get("gold_step_explanation", ""),
        }
        examples.append(example)
    
    return examples
