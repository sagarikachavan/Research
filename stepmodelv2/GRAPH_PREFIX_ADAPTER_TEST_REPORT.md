# Graph Prefix Adapter Test Report

## Research Question

Does the LLM understand graph structure when graph embeddings are converted to soft prompt tokens via the Graph Prefix Adapter in stepmodelv2?

## Background

The stepmodelv2 pipeline uses a Graph Prefix Adapter to convert graph embeddings (256-dim) from a Stage-1 GNN into 8 soft prompt tokens (8 × 3584-dim) that are prepended to the LLM input. The goal is to enable the LLM to understand and reason about graph structure for step prediction tasks.

However, it was unclear whether the LLM actually understood the graph structure encoded in these soft prompt tokens, or if it was just using them as a conditioning signal without explicit graph understanding.

## Methodology

### Test Task: Adjacency Prediction

To test graph structure understanding, we designed a simple but effective test: given a node ID from the graph, can the LLM predict which nodes are directly connected (adjacent) to it?

This is a fundamental graph reasoning task that requires understanding:
- Node identities and relationships
- Graph connectivity structure
- Edge connections between specific nodes

### Architecture

The test uses the same architecture as Stage 2:

1. **Stage-1 GNN Encoder** (frozen): Processes graph structure and produces 256-dim graph embedding
2. **Graph Prefix Adapter**: Projects 256-dim → 8 soft prompt tokens (8 × 3584-dim)
3. **Qwen2.5-7B-Instruct + LoRA**: LLM that processes soft prompt tokens + text prompts

### Test Setup

- **Graph**: active_graph.json (35 nodes, 45 edges)
- **Test nodes**: 5 representative nodes from the graph
- **Evaluation metric**: Recall (percentage of correct adjacent nodes identified)
- **Comparison**: Untrained vs Stage 2 trained vs graph structure trained weights

## Scripts

### test_graph_prefix_adapter.py

Test script to evaluate whether the LLM understands graph structure via soft prompt tokens.

**Functionality**:
- Loads graph from JSON and builds torch_geometric Data object
- Uses frozen Stage-1 GNN encoder to generate graph embedding
- Converts graph embedding to 8 soft prompt tokens via Graph Prefix Adapter
- Prepends soft prompt tokens to LLM input embeddings
- Queries LLM to predict adjacent nodes for test nodes
- Compares predictions against ground truth adjacency

**Modes**:
- `--mode untrained`: Test with random weights (baseline)
- `--mode trained`: Test with trained weights
- `--mode both`: Compare both modes
- `--checkpoint`: Specify checkpoint directory

### train_graph_structure.py

Training script to train the Graph Prefix Adapter specifically on graph structure tasks.

**Problem**: Stage 2 training (stage2_sft_qwen.py) focused on step prediction and did not teach the LLM to understand graph structure explicitly.

**Solution**: Train on explicit graph structure tasks to teach the model to encode and decode graph information.

**Supported Tasks**:
- Adjacency prediction: Given a node ID, predict directly connected nodes
- Node type prediction: Predict node type (Agent/Search/Track)
- Edge type prediction: Predict edge type (StateTransition/SearchUpdate/TrackUpdate/Prediction)
- Path prediction: Predict 2-hop nodes in the graph

**Training Setup**:
- Loads all graph JSON files from processed_data/train directory
- Builds task-specific training samples from graph structure
- Trains Graph Prefix Adapter + Qwen + LoRA on target task
- Saves trained weights to checkpoints/graph_structure/

## Results

### Baseline Tests

**Untrained weights (random initialization)**:
- Average Recall: 0%
- Behavior: LLM hallucinates fake node IDs (e.g., `agent:active:ACTION`, `search:active:r1_0`)
- Conclusion: Random weights provide no graph structure understanding

**Stage 2 trained weights**:
- Average Recall: 0%
- Behavior: LLM outputs empty lists with explanations that no adjacent nodes exist
- Conclusion: Stage 2 training (step prediction) did not teach graph structure understanding

### Graph Structure Training

Created dedicated training script and trained specifically on adjacency prediction:

- **Training data**: 175 graphs, 6,144 adjacency samples
- **Training epochs**: 5
- **Final training loss**: 0.0922
- **Checkpoint directory**: checkpoints/graph_structure/

### Final Test Results with Graph Structure Training

**Command**: `python test.py --mode trained --checkpoint checkpoints/graph_structure`

**Test Results**:

Test 1: agent:active:START
- Ground truth: ['agent:active:r1_base', 'search:active:r1_base']
- LLM prediction: agent:active:r1_base, search:active:r1_base
- Recall: 100.00%

Test 2: search:active:r1_base
- Ground truth: ['agent:active:START', 'track:active:r1_base']
- LLM prediction: track:active:r1_base, agent:active:START
- Recall: 100.00%

Test 3: track:active:r1_base
- Ground truth: ['search:active:r1_base', 'agent:active:r1_base']
- LLM prediction: search:active:r1_base, agent:active:r1_base
- Recall: 100.00%

Test 4: agent:active:r1_base
- Ground truth: ['agent:active:START', 'track:active:r1_base', 'agent:active:r1_s1_1.6', 'search:active:r1_s1_1.6']
- LLM prediction: track:active:r1_base, agent:active:START, search:active:r1_s1_1.3.5
- Recall: 50.00%

Test 5: search:active:r1_s1_1.6
- Ground truth: ['agent:active:r1_base', 'track:active:r1_s1_1.6']
- LLM prediction: track:active:r1_s1_1.6, agent:active:r1_base
- Recall: 100.00%

**Overall Results**:
- Average Recall: 90.00%
- Individual: 100%, 100%, 100%, 50%, 100%

## Key Findings

1. **Architecture works when trained appropriately**: The Graph Prefix Adapter successfully encodes graph structure when trained on the right task
2. **Task-specific training is essential**: Stage 2 training (step prediction) did not teach graph structure understanding (0% recall)
3. **Dedicated training succeeds**: Training specifically on adjacency prediction achieved 90% recall
4. **Soft prompt tokens contain meaningful information**: The 8 soft prompt tokens successfully encode graph structure when trained appropriately
5. **LLM can decode graph information**: The LLM can learn to extract and utilize graph structure from soft prompt tokens

## Conclusion

The Graph Prefix Adapter architecture is effective for enabling LLMs to understand graph structure, but requires task-specific training. The Stage 2 training focused on step prediction rather than explicit graph structure understanding, resulting in 0% recall on adjacency prediction. Dedicated graph structure training successfully teaches the model to encode and utilize graph information for reasoning tasks, achieving 90% recall.

This demonstrates that:
- The LLM can learn to understand soft prompt tokens when trained on graph structure tasks
- The Graph Prefix Adapter successfully encodes graph structure in a way the LLM can decode and use for reasoning
- Different training objectives require different training strategies
- The 8 soft prompt tokens contain meaningful graph structure information when trained appropriately

## Files Created

- `test_graph_prefix_adapter.py`: Test script with trained/untrained comparison and checkpoint loading
- `train_graph_structure.py`: Training script for graph structure tasks (adjacency, node type, edge type, path prediction)
- `checkpoints/graph_structure/`: Trained weights for adjacency prediction (90% recall)
- `GRAPH_PREFIX_ADAPTER_TEST_REPORT.md`: This comprehensive documentation

## Next Steps

- Test other graph structure tasks (node type, edge type, path prediction)
- Evaluate impact of graph structure training on original step prediction task
- Consider multi-task training combining step prediction with graph structure understanding
- Investigate whether graph structure training improves overall model performance
