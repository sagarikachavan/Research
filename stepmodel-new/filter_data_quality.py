#!/usr/bin/env python3
"""
Data Quality Filtering for stepmodel-new

This script filters the expanded dataset to remove low-quality samples that may be
hurting model performance. It uses multiple heuristics to assess data quality:
- Text length consistency
- Label distribution balance
- Graph structure quality
- Embedding quality metrics
"""

import json
import os
import argparse
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def assess_text_quality(sample: Dict) -> float:
    """Assess text quality based on length and structure."""
    score = 1.0
    
    # Check text length
    if 'text' in sample:
        text_len = len(sample['text'])
        if text_len < 50:  # Too short
            score *= 0.5
        elif text_len > 10000:  # Too long
            score *= 0.8
    
    # Check for required fields
    required_fields = ['step_label', 'mcp_label', 'graph_data']
    for field in required_fields:
        if field not in sample:
            score *= 0.3
    
    return score


def assess_label_quality(sample: Dict) -> float:
    """Assess label quality based on consistency and validity."""
    score = 1.0
    
    # Check step label validity
    if 'step_label' in sample:
        step_label = sample['step_label']
        if not isinstance(step_label, int) or step_label < 0:
            score *= 0.5
    
    # Check MCP label validity
    if 'mcp_label' in sample:
        mcp_label = sample['mcp_label']
        if not isinstance(mcp_label, int) or mcp_label not in [0, 1]:
            score *= 0.5
    
    return score


def assess_graph_quality(sample: Dict) -> float:
    """Assess graph structure quality."""
    score = 1.0
    
    if 'graph_data' not in sample:
        return 0.3
    
    graph = sample['graph_data']
    
    # Check for nodes and edges
    if 'nodes' not in graph or 'edges' not in graph:
        score *= 0.5
    else:
        num_nodes = len(graph['nodes'])
        num_edges = len(graph['edges'])
        
        # Basic sanity checks
        if num_nodes == 0:
            score *= 0.3
        if num_edges == 0 and num_nodes > 1:
            score *= 0.5
        if num_edges > num_nodes * (num_nodes - 1) / 2:  # More edges than possible
            score *= 0.3
    
    return score


def assess_embedding_quality(sample: Dict) -> float:
    """Assess embedding quality based on dimensions and values."""
    score = 1.0
    
    if 'node_embeddings' not in sample and 'edge_embeddings' not in sample:
        return 0.5
    
    # Check embedding dimensions
    for emb_key in ['node_embeddings', 'edge_embeddings']:
        if emb_key in sample:
            embeddings = sample[emb_key]
            if not isinstance(embeddings, list) or len(embeddings) == 0:
                score *= 0.5
            else:
                # Check for NaN or infinite values
                try:
                    emb_array = np.array(embeddings)
                    if np.any(np.isnan(emb_array)) or np.any(np.isinf(emb_array)):
                        score *= 0.3
                except:
                    score *= 0.5
    
    return score


def calculate_overall_quality(sample: Dict, weights: Dict[str, float] = None) -> float:
    """Calculate overall quality score for a sample."""
    if weights is None:
        weights = {
            'text': 0.3,
            'label': 0.3,
            'graph': 0.2,
            'embedding': 0.2
        }
    
    text_score = assess_text_quality(sample)
    label_score = assess_label_quality(sample)
    graph_score = assess_graph_quality(sample)
    embedding_score = assess_embedding_quality(sample)
    
    overall_score = (
        weights['text'] * text_score +
        weights['label'] * label_score +
        weights['graph'] * graph_score +
        weights['embedding'] * embedding_score
    )
    
    return overall_score


def filter_dataset(input_file: str, output_file: str, threshold: float = 0.7, 
                   max_samples: int = None) -> Tuple[int, int]:
    """
    Filter dataset based on quality threshold.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output filtered JSON file
        threshold: Quality threshold (0-1)
        max_samples: Maximum number of samples to keep
    
    Returns:
        Tuple of (original_count, filtered_count)
    """
    print(f"Loading data from {input_file}...")
    data = load_json_data(input_file)
    original_count = len(data)
    print(f"Original dataset size: {original_count}")
    
    print("Assessing data quality...")
    quality_scores = []
    for sample in data:
        score = calculate_overall_quality(sample)
        quality_scores.append(score)
    
    # Filter based on threshold
    filtered_data = []
    for sample, score in zip(data, quality_scores):
        if score >= threshold:
            sample['quality_score'] = score
            filtered_data.append(sample)
    
    # Sort by quality score and optionally limit
    filtered_data.sort(key=lambda x: x['quality_score'], reverse=True)
    if max_samples and len(filtered_data) > max_samples:
        filtered_data = filtered_data[:max_samples]
    
    filtered_count = len(filtered_data)
    print(f"Filtered dataset size: {filtered_count}")
    print(f"Removed {original_count - filtered_count} samples ({(original_count - filtered_count)/original_count*100:.1f}%)")
    
    # Print quality statistics
    if quality_scores:
        print(f"Quality score statistics:")
        print(f"  Mean: {np.mean(quality_scores):.3f}")
        print(f"  Median: {np.median(quality_scores):.3f}")
        print(f"  Std: {np.std(quality_scores):.3f}")
        print(f"  Min: {np.min(quality_scores):.3f}")
        print(f"  Max: {np.max(quality_scores):.3f}")
    
    # Save filtered data
    print(f"Saving filtered data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Remove quality_score from samples before saving
    for sample in filtered_data:
        if 'quality_score' in sample:
            del sample['quality_score']
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print("Filtering complete!")
    return original_count, filtered_count


def main():
    parser = argparse.ArgumentParser(description='Filter dataset based on quality')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Quality threshold (0-1, default: 0.7)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to keep')
    
    args = parser.parse_args()
    
    filter_dataset(args.input, args.output, args.threshold, args.max_samples)


if __name__ == '__main__':
    main()
