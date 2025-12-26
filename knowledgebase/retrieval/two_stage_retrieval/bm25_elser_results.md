# Two-Stage BM25 + ELSER Retrieval Results

## Overview

This document presents results from a two-stage retrieval approach:
1. **Stage 1**: BM25 lexical retrieval (local, fast) retrieves top-k candidates
2. **Stage 2**: ELSER neural reranking (via Elasticsearch) reranks to top 10

**Source**: `scripts/ideas/retrieval_tasks/bm25_bge_rerank/`

---

## Method

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: BM25 Retrieval (Local)                                 │
│   - PyTerrier BM25 implementation                               │
│   - Retrieve top-k candidates (k=50, 200, 500)                  │
│   - Fast lexical matching, no network required                  │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: ELSER Reranking (Elasticsearch Cloud)                  │
│   - Filter query on BM25 candidate doc IDs                      │
│   - ELSER text_expansion scoring on filtered set                │
│   - Return top 10 by ELSER score                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Main Results: BM25@500 + ELSER (Rewrite Query)

### nDCG@10 Comparison (All Methods)

| Domain | BM25@500+ELSER | BM25@200+ELSER | BM25@50+ELSER | Pure ELSER | Pure BGE | Pure BM25 |
|--------|----------------|----------------|---------------|------------|----------|-----------|
| clapnq | **0.563**      | 0.554          | 0.512         | 0.578      | 0.498    | 0.294     |
| cloud  | **0.424**      | 0.414          | 0.378         | 0.438      | 0.342    | 0.233     |
| fiqa   | **0.403**      | 0.379          | 0.337         | 0.436      | 0.341    | 0.174     |
| govt   | **0.505**      | 0.488          | 0.460         | 0.517      | 0.420    | 0.342     |

### Recall@10 Comparison

| Domain | BM25@500+ELSER | BM25@200+ELSER | BM25@50+ELSER |
|--------|----------------|----------------|---------------|
| clapnq | 0.669          | 0.654          | 0.585         |
| cloud  | 0.509          | 0.487          | 0.452         |
| fiqa   | 0.479          | 0.439          | 0.376         |
| govt   | 0.628          | 0.613          | 0.566         |

---

## Gap to Pure ELSER

### BM25@500+ELSER vs Pure ELSER

| Domain | BM25@500+ELSER | Pure ELSER | Gap         |
|--------|----------------|------------|-------------|
| clapnq | 0.563          | 0.578      | **-2.6%**   |
| cloud  | 0.424          | 0.438      | **-3.2%**   |
| fiqa   | 0.403          | 0.436      | -7.6%       |
| govt   | 0.505          | 0.517      | **-2.3%**   |
| **Avg**| -              | -          | **-3.9%**   |

**Key Finding**: With BM25@500, the gap to pure ELSER is now:
- **2-3%** for ClapNQ, Cloud, Govt (nearly matching pure ELSER!)
- **7.6%** for FiQA (vocabulary mismatch remains a challenge)

---

## Impact of BM25 Pool Size

### Improvement by Pool Size (nDCG@10)

| Domain | @50→@200 | @200→@500 | Total @50→@500 |
|--------|----------|-----------|----------------|
| clapnq | +8.2%    | +1.5%     | **+10.0%**     |
| cloud  | +9.5%    | +2.3%     | **+12.2%**     |
| fiqa   | +12.4%   | +6.5%     | **+19.6%**     |
| govt   | +6.0%    | +3.6%     | **+9.8%**      |
| **Avg**| +9.0%    | +3.5%     | **+12.9%**     |

### Diminishing Returns Analysis

| Transition | Avg Improvement | Notes |
|------------|-----------------|-------|
| @50 → @200 | **+9.0%** | Largest gains |
| @200 → @500 | **+3.5%** | Diminishing returns |
| @500 → @1000 | ~1-2% (est.) | Further diminishing |

**Insight**: The sweet spot is likely around **BM25@200-500**. Beyond 500, gains are minimal while computational cost increases.

---

## Comparison: Two-Stage Rerankers

### BM25@50 + Different Rerankers (nDCG@10, rewrite query)

| Domain | BM25@50+ELSER | BM25@50+BGE | Δ ELSER vs BGE |
|--------|---------------|-------------|----------------|
| clapnq | **0.512**     | 0.461       | +11.1%         |
| cloud  | **0.378**     | 0.345       | +9.6%          |
| fiqa   | **0.337**     | 0.298       | +13.1%         |
| govt   | **0.460**     | 0.413       | +11.4%         |
| **Avg**| -             | -           | **+11.3%**     |

**Finding**: ELSER outperforms BGE as a reranker by ~11% on average.

---

## Full Results Table: BM25@500 + ELSER (All k values)

### ClapNQ

| Metric    | @1     | @3     | @5     | @10    |
|-----------|--------|--------|--------|--------|
| nDCG      | 0.519  | 0.467  | 0.507  | 0.563  |
| Recall    | 0.210  | 0.422  | 0.542  | 0.669  |

### Cloud

| Metric    | @1     | @3     | @5     | @10    |
|-----------|--------|--------|--------|--------|
| nDCG      | 0.383  | 0.350  | 0.376  | 0.424  |
| Recall    | 0.177  | 0.333  | 0.401  | 0.509  |

### FiQA

| Metric    | @1     | @3     | @5     | @10    |
|-----------|--------|--------|--------|--------|
| nDCG      | 0.383  | 0.329  | 0.353  | 0.403  |
| Recall    | 0.160  | 0.289  | 0.365  | 0.479  |

### Govt

| Metric    | @1     | @3     | @5     | @10    |
|-----------|--------|--------|--------|--------|
| nDCG      | 0.413  | 0.407  | 0.446  | 0.505  |
| Recall    | 0.192  | 0.396  | 0.494  | 0.628  |

---

## Conclusions

1. **BM25@500+ELSER nearly matches pure ELSER** (within 2-3% on 3/4 domains)

2. **Pool size matters**: 
   - @50→@500 yields +13% average improvement
   - Diminishing returns beyond @500

3. **ELSER is a stronger reranker than BGE** (+11% improvement)

4. **FiQA remains challenging** (-7.6% gap) due to financial vocabulary mismatch

5. **Trade-offs**:
   | Aspect | Two-Stage BM25@500+ELSER | Pure ELSER |
   |--------|--------------------------|------------|
   | Quality | 96-97% of ELSER | 100% |
   | ES API calls | Fewer (filter only) | Full corpus |
   | Local compute | BM25 stage | None |
   | Latency | Potentially lower | Baseline |

---

## Recommendations

1. **For production with quality priority**: Use pure ELSER

2. **For reduced API load / latency-sensitive**: Use BM25@500+ELSER
   - Only 3-4% quality tradeoff
   - Significant reduction in ES scoring workload

3. **For FiQA domain**: Consider alternative first-stage (semantic BM25, hybrid) to improve vocabulary matching

4. **Optimal pool size**: **BM25@200-500** balances quality vs compute

---

## Source Code

- Main script: `scripts/ideas/retrieval_tasks/bm25_bge_rerank/run_bm25_elser_rerank.py`
- Results: `scripts/ideas/retrieval_tasks/bm25_bge_rerank/results/bm25_elser_*.csv`

---

## Change Log

| Date | Change |
|------|--------|
| 2024-12-25 | Initial results with BM25@50 and @200 |
| 2024-12-25 | Added BM25@500 results, updated conclusions |
