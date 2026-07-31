# Graph Prefix Adapter Test Report

## Objective
Test whether the LLM understands graph structure when graph embeddings are converted to soft prompt tokens via the Graph Prefix Adapter in stepmodelv2.

## Methodology
- **Test Task**: Adjacency prediction - given a node ID, predict directly connected nodes
- **Architecture**: Graph Prefix Adapter (256-dim → 8 soft prompt tokens) + Qwen2.5-7B-Instruct + LoRA
- **Test Cases**: 5 nodes from the active machine graph (35 nodes, 45 edges total)
- **Evaluation Metric**: Recall (percentage of correct adjacent nodes identified)

## Results

### Baseline Tests
- **Untrained weights**: 0% recall (hallucinated fake node IDs)
- **Stage 2 trained weights**: 0% recall (conservative empty responses)

### Graph Structure Training
Created dedicated training script (`train_graph_structure.py`) to train specifically on graph structure tasks:
- **Training data**: 175 graphs, 6,144 adjacency samples
- **Training epochs**: 5
- **Final training loss**: 0.0922

### Final Results with Graph Structure Training
- **Average Recall**: 90%
- **Individual test results**: 100%, 100%, 100%, 50%, 100%

## Key Findings

1. **Architecture works**: The Graph Prefix Adapter successfully encodes graph structure when trained appropriately
2. **Task-specific training required**: Stage 2 training (step prediction) did not teach graph structure understanding
3. **Dedicated training succeeds**: Training specifically on adjacency prediction achieved 90% recall
4. **Soft prompt tokens effective**: The 8 soft prompt tokens contain meaningful graph structure information

## Conclusion
The Graph Prefix Adapter architecture is effective for enabling LLMs to understand graph structure, but requires task-specific training. The Stage 2 training focused on step prediction rather than explicit graph structure understanding. Dedicated graph structure training successfully teaches the model to encode and utilize graph information for reasoning tasks.

## Files Created
- `test_graph_prefix_adapter.py`: Test script with trained/untrained comparison
- `train_graph_structure.py`: Training script for graph structure tasks
- `checkpoints/graph_structure/`: Trained weights for adjacency prediction

## Next Steps
- Test other graph structure tasks (node type, edge type, path prediction)
- Evaluate impact on original step prediction task
- Consider multi-task training combining step prediction with graph structure understanding
