# DH-RAG: Dynamic Historical RAG

Implementation of the DH-RAG paper for multi-turn conversational retrieval.

## The Original Paper

**DH-RAG** (Dynamic Historical RAG) is from a 2024 paper that addresses the problem of **irrelevant history noise** in multi-turn conversational retrieval.

The key insight is that in long conversations, simply including all previous turns (or the last N turns) can hurt retrieval performance because:

- Conversations often **switch topics** mid-stream
- Earlier turns may be completely irrelevant to the current query
- Including noisy context confuses both query rewriting and retrieval

**Reference**: DH-RAG: A Dynamic Historical RAG for Multi-Turn Conversations (2024)

## Core Paper Concepts

The paper introduces a **3-tier hierarchical structure** for organizing conversation history:

```
                    ┌────────────────────┐
                    │   Cluster Layer    │  (Topic-level grouping)
                    │  k clusters        │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Summary 1   │     │   Summary 2   │     │   Summary m   │  (Sub-topic representatives)
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
    ┌───┴───┐             ┌───┴───┐             ┌───┴───┐
    ▼       ▼             ▼       ▼             ▼       ▼
┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐
│Triple│ │Triple│     │Triple│ │Triple│     │Triple│ │Triple│  (Query, Passage, Response)
└──────┘ └──────┘     └──────┘ └──────┘     └──────┘ └──────┘
```

### Paper Components

1. **Dynamic Historical Information Database**: `H = {(q, p, r)₁, (q, p, r)₂, ..., (q, p, r)ₜ₋₁}`
   - Each triple stores: user query, retrieved passage, agent response

2. **Historical Query Clustering**: Groups triples by topic using TF-IDF + K-Means

3. **Hierarchical Matching**: TF-IDF-based tree navigation (Cluster → Summary → Triple)

4. **Chain-of-Thought Tracking**: Detects sequences of related queries
   - Paper reports: Average chain length = 1.73 steps
   - Distribution: 1-2 steps (32%), 2-3 steps (40%), 3-4 steps (21%), 4-5 steps (6%)

5. **α-Weighted Scoring**: Balances semantic relevance vs. recency

## Our Implementation

### Scoring Formula

```
Score = (α × relevance) + ((1-α) × recency) + cluster_bonus + summary_bonus + chain_bonus
```

Where:

- **relevance**: Cosine similarity between query embedding and history turn embedding
- **recency**: Normalized position in history (0-1, higher = more recent)
- **cluster_bonus**: +0.10 if the turn is in the matched topic cluster
- **summary_bonus**: +0.05 if the turn is in the matched summary
- **chain_bonus**: +0.05 if the turn is part of an active chain-of-thought

### Pipeline

1. Build DH-RAG database from conversation history
2. Retrieve top-k relevant history using hierarchical matching
3. Pass selected history to Claude for query rewriting
4. Save rewritten query in BEIR format

## Files

- `dh_rag_lite.py` - Core DH-RAG implementation
- `run_dh_rag_rewrite.py` - Pipeline for query rewriting across datasets
- `run_retrieval.py` - Run retrieval with rewritten queries
- `evaluate_and_compare.py` - Evaluate and compare results
- `run_experiment.sh` - Full experiment runner

## Parameters

| Parameter         | Default                    | Description                                      |
| ----------------- | -------------------------- | ------------------------------------------------ |
| `alpha`           | 0.6                        | Relevance vs recency weight (1 = pure relevance) |
| `max_clusters`    | 5                          | Maximum topic clusters                           |
| `chain_threshold` | 0.4                        | Similarity threshold for chain detection         |
| `top_k`           | 3                          | Number of history turns to retrieve              |
| `embedding_model` | `all-MiniLM-L6-v2`         | Sentence transformer model                       |
| `llm_model`       | `claude-sonnet-4-5-20250929` | LLM for query rewriting                          |

## Comparison with Other Approaches

| Approach             | History Selection         | Hierarchy   | Content-Aware      |
| -------------------- | ------------------------- | ----------- | ------------------ |
| **Naive (Last-N)**   | Most recent N turns       | None        | No                 |
| **Targeted Rewrite** | Similarity threshold      | Flat        | Embedding only     |
| **DH-RAG**           | Cluster + Summary + Chain | 3-tier tree | TF-IDF + Embedding |

## Expected Benefits

DH-RAG should outperform simpler methods on:

- **Topic-switching conversations**: User returns to an earlier topic
- **Long conversations**: Where naive Last-N includes irrelevant recent turns
- **Chain-of-thought queries**: Follow-up questions that build on earlier related turns

Minimal benefit expected for:

- **Turn 1 queries**: No history to filter
- **Short conversations** (2-3 turns): All context is typically relevant
