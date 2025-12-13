# DH-RAG: Dynamic Historical RAG

Implementation of the DH-RAG paper for multi-turn conversational retrieval.

## Overview

DH-RAG addresses the problem of **irrelevant history noise** in multi-turn conversations by:

1. **Dynamic History Database**: Stores (query, passage, response) triples with embeddings
2. **3-Tier Hierarchical Structure**: Clusters → Summaries → Triples
3. **TF-IDF Hierarchical Matching**: Efficiently navigates history tree
4. **Chain-of-Thought Tracking**: Detects sequences of related queries
5. **α-Weighted Scoring**: Balances relevance vs. recency

## Key Formula

```
Score = α × Relevance + (1-α) × Recency + ClusterBonus + SummaryBonus + ChainBonus
```

## Files

- `dh_rag_lite.py` - Core DH-RAG implementation
- `run_dh_rag_rewrite.py` - Pipeline for query rewriting across datasets
- `run_experiment.sh` - Full experiment runner

## Usage

### Single Conversation

```python
from dh_rag_lite import DHRAG

dh_rag = DHRAG(alpha=0.6, max_clusters=5)

# Add history
dh_rag.add_interaction("What is Python?", "Python is a programming language...")
dh_rag.add_interaction("How do I install it?", "Download from python.org...")

# Retrieve relevant history for new query
results = dh_rag.retrieve("What IDE should I use for Python?", top_k=3)
```

### Full Dataset

```bash
# Run with defaults
./run_experiment.sh

# Or with custom config
ALPHA=0.7 TOP_K=5 DOMAINS="clapnq" ./run_experiment.sh

# Or directly
python run_dh_rag_rewrite.py --domains clapnq cloud --alpha 0.6 --top_k 3
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.6 | Relevance vs recency weight (0=recency only, 1=relevance only) |
| `max_clusters` | 5 | Maximum topic clusters |
| `chain_threshold` | 0.4 | Similarity threshold for chain detection |
| `top_k` | 3 | Number of history turns to retrieve |

## Comparison with Targeted Rewrite

| Approach | History Selection | Hierarchy |
|----------|------------------|-----------|
| **Naive (Last-N)** | Most recent N turns | None |
| **Targeted Rewrite** | Similarity threshold | Flat |
| **DH-RAG** | Cluster + Summary + Chain matching | 3-tier tree |

## Expected Benefits

DH-RAG should outperform baselines on:
- Conversations with **topic switches** (user returns to earlier topic)
- Long conversations where naive Last-N includes irrelevant turns
- Queries that require context from earlier, semantically related turns

## Reference

Based on: DH-RAG: A Dynamic Historical RAG for Multi-Turn Conversations (2024)

