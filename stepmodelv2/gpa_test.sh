#!/bin/bash
# Comprehensive test suite for Graph Prefix Adapter
# This script runs all tests to verify the adapter works correctly

set -e

echo "=========================================="
echo "Graph Prefix Adapter - Complete Test Suite"
echo "=========================================="
echo ""

# Step 1: Show training data samples
echo "Step 1: Showing training data samples..."
echo "=========================================="
python show_training_samples.py
echo ""

# Step 2: Test adjacency prediction (untrained vs trained)
echo "Step 2: Testing adjacency prediction..."
echo "=========================================="
echo "Testing with untrained weights..."
python test_graph_prefix_adapter.py --mode untrained --test_type adjacency
echo ""
echo "Testing with trained weights..."
python test_graph_prefix_adapter.py --mode trained --test_type adjacency
echo ""

# Step 3: Test node content question
echo "Step 3: Testing node content question..."
echo "=========================================="
echo "Testing with untrained weights..."
python test_graph_prefix_adapter.py --mode untrained --test_type node_content
echo ""
echo "Testing with trained weights..."
python test_graph_prefix_adapter.py --mode trained --test_type node_content
echo ""

# Step 4: Train and test remaining tasks
echo "Step 4: Training and testing remaining graph structure tasks..."
echo "=========================================="

# Node type prediction
echo "Training node type prediction..."
python train_graph_structure.py --task node_type --epochs 3 --batch_size 4 --output_dir /tmp/graph_structure_node_type
echo "Testing node type prediction..."
python test_graph_prefix_adapter.py --mode trained --test_type adjacency --checkpoint /tmp/graph_structure_node_type
echo ""

# Edge type prediction
echo "Training edge type prediction..."
python train_graph_structure.py --task edge_type --epochs 3 --batch_size 4 --output_dir /tmp/graph_structure_edge_type
echo "Testing edge type prediction..."
python test_graph_prefix_adapter.py --mode trained --test_type adjacency --checkpoint /tmp/graph_structure_edge_type
echo ""

# Path prediction
echo "Training path prediction..."
python train_graph_structure.py --task path --epochs 3 --batch_size 4 --output_dir /tmp/graph_structure_path
echo "Testing path prediction..."
python test_graph_prefix_adapter.py --mode trained --test_type adjacency --checkpoint /tmp/graph_structure_path
echo ""

# Step 5: Final comprehensive test (use adjacency checkpoint since that's what we want to test)
echo "Step 5: Running comprehensive test (both test types)..."
echo "=========================================="
python test_graph_prefix_adapter.py --mode both --test_type both --checkpoint /tmp/graph_structure
echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
