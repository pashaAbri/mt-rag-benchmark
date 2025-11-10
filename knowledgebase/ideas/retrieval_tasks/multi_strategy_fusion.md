# Multi-Strategy Retrieval Fusion

## Overview

Instead of selecting a single "best" query strategy, combine results from multiple strategies to improve recall and ranking quality. Each strategy captures different aspects of information need, and their combination can surface more relevant documents.

**Core Principle:** Different query formulations retrieve different (but potentially relevant) documents. Fusion maximizes coverage.

---

## Motivation

Current MT-RAG benchmark evaluates strategies **independently**:
- Last Turn: R@5 = 0.44, nDCG@5 = 0.41
- Rewrite: R@5 = 0.48, nDCG@5 = 0.44
- Pure Extractive: R@5 = 0.42, nDCG@5 = 0.39

**Problem:** Each strategy may retrieve non-overlapping relevant documents. Picking one strategy misses documents found by others.

**Solution:** Retrieve with all strategies, pool candidates, rerank using fusion.

---

## Method: Reciprocal Rank Fusion (RRF)

### Why RRF?

- **Robust**: Works across different retriever types (BM25, BGE, ELSER)
- **Score-agnostic**: Uses ranks, not raw scores (avoids score scale issues)
- **Parameter-free**: No tuning needed (k=60 is standard)
- **Proven**: Used in Elasticsearch, RAG Fusion, production systems

### RRF Formula

```
score(doc) = Σ 1/(k + rank_i)
            i=1..N
```

Where:
- `rank_i` = position of document in strategy i's results (1-indexed)
- `k` = constant (default: 60, from Cormack et al. SIGIR 2009)
- `N` = number of strategies

**Intuition:** Documents ranked highly in multiple strategies get highest scores.

### Example

```
Strategy 1 (lastturn):   [doc_A(rank=1), doc_B(rank=2), doc_C(rank=5)]
Strategy 2 (rewrite):    [doc_B(rank=1), doc_C(rank=3), doc_A(rank=8)]
Strategy 3 (extractive): [doc_A(rank=2), doc_D(rank=1), doc_C(rank=4)]

RRF scores (k=60):
doc_A: 1/61 + 1/68 + 1/62 = 0.0472  ← Consistent across all
doc_C: 1/65 + 1/63 + 1/64 = 0.0469  ← Consistent across all
doc_B: 1/62 + 1/61 + 0    = 0.0325  ← Missing in one strategy
doc_D: 0    + 0    + 1/61 = 0.0164  ← Only in one strategy

Final ranking: [doc_A, doc_C, doc_B, doc_D]
```

---

## Architecture

```
┌─────────────────────┐
│  Conversation       │
│  + Current Query    │
└──────────┬──────────┘
           │
           ├────────────────┬────────────────┐
           │                │                │
           ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐    ┌──────────┐
    │ Strategy │     │ Strategy │    │ Strategy │
    │    1     │     │    2     │    │    3     │
    │(lastturn)│     │(rewrite) │    │(extract) │
    └─────┬────┘     └─────┬────┘    └─────┬────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐    ┌──────────┐
    │Retriever │     │Retriever │    │Retriever │
    │ top-10   │     │ top-10   │    │ top-10   │
    └─────┬────┘     └─────┬────┘    └─────┬────┘
          │                │                │
          └────────┬───────┴────────────────┘
                   ▼
          ┌─────────────────┐
          │  Document Pool  │
          │   (15-30 docs)  │
          │  + Deduplicate  │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │  RRF Reranking  │
          │  (rank fusion)  │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │   Top-K Final   │
          │   (typically 10)│
          └─────────────────┘
```

---

## Expected Benefits

Based on MQRF-RAG (Yang et al. 2025) and RAG Fusion literature:

**Recall Improvement:**
- +3-8% Recall@5 (more relevant docs in top-5)
- Better coverage of information needs

**Ranking Improvement:**
- +2-5% nDCG@5 (better ordering of relevant docs)
- Documents appearing in multiple strategies rank higher

**Robustness:**
- Reduces impact of poor single-strategy performance
- Works across domains (clapnq, cloud, fiqa, govt)
- Works across retrievers (BM25, BGE, ELSER)

---

## Implementation Plan

### Phase 1: RRF Fusion Script (Standalone)

**Input:** Multiple retrieval result files (JSONL format)
```json
{"query_id": "123", "results": {"doc1": 0.95, "doc2": 0.87, ...}}
```

**Output:** Fused results (same format)
```json
{"query_id": "123", "results": {"doc1": 0.047, "doc2": 0.032, ...}}
```

**Components:**
1. Load retrieval results from multiple files
2. Apply RRF fusion
3. Keep top-K documents
4. Save in standard format for evaluation

### Phase 2: Evaluation

Run existing evaluation scripts on fused results:
```bash
python scripts/evaluation/run_retrieval_eval.py \
  --results fusion/elser_clapnq_fusion.jsonl \
  --qrels human/retrieval_tasks/clapnq/qrels/dev.tsv \
  --output fusion/elser_clapnq_fusion_evaluated.csv
```

### Phase 3: Comparison

Compare performance:
```
| Strategy      | R@5  | nDCG@5 | Improvement |
|---------------|------|--------|-------------|
| Last Turn     | 0.44 | 0.41   | baseline    |
| Rewrite       | 0.48 | 0.44   | +9% / +7%   |
| Extractive    | 0.42 | 0.39   | -5% / -5%   |
| Fusion (All)  | 0.51?| 0.47?  | +16%? / +15%? |
```

---

## Strategy Combinations to Test

1. **Basic Fusion** (2 strategies):
   - `lastturn + rewrite`
   - `rewrite + extractive`

2. **Full Fusion** (3 strategies):
   - `lastturn + rewrite + extractive`
   - `lastturn + rewrite + hybrid_extractive`

3. **Domain-Specific** (if some strategies work better per domain):
   - ClapNQ: `rewrite + extractive`
   - FiQA: `lastturn + rewrite`
   - etc.

---

## Alternative Fusion Methods (Future Work)

### Score Normalization + Weighted Sum
- Normalize scores to [0,1], combine with weights
- Requires tuning weights per domain/retriever
- Only works when scores are comparable

### Cross-Encoder Reranking
- Use neural model (e.g., `BAAI/bge-reranker-base`)
- Higher quality but slower (50-200ms per doc on CPU)
- Requires loading additional model

### Weighted RRF
- `score(doc) = Σ w_i / (k + rank_i)` with strategy-specific weights
- Adds tuning complexity
- May overfit to dev set

---

## References

- **Cormack et al.** "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.
- **Rackauckas.** "RAG Fusion: Better RAG with reciprocal reranking." 2023.
- **Yang et al.** "MQRF-RAG: Multi-Strategy Query Rewriting Framework for RAG." 2025.
- **Elasticsearch** Hybrid Search: Uses RRF for combining BM25 + kNN results.

---

## Key Decisions

✅ **Use RRF** (not score normalization): Works across different retriever types  
✅ **Start with 3-way fusion**: lastturn + rewrite + extractive  
✅ **Use k=60**: Standard value from literature  
✅ **Keep top-10**: Match existing evaluation setup  
✅ **Reuse existing retrieval results**: No need to re-run retrievers

---

## Success Criteria

**Minimum Viable Success:**
- Fusion outperforms best single strategy by ≥2% R@5

**Strong Success:**
- Fusion outperforms best single strategy by ≥5% R@5
- Consistent improvement across all 4 domains

**Publication-Worthy:**
- Fusion consistently best across domains and retrievers
- Clear analysis of which strategy combinations work best
- Insights on when fusion helps most (query characteristics, domain type)

