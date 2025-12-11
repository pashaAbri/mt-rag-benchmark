# Reciprocal Rank Fusion (RRF) Experiments

## Objective

To evaluate whether **Reciprocal Rank Fusion (RRF)**—a computationally cheap, rank-based fusion method—can combine diverse retrieval results to outperform the best single retrieval baseline. This serves as a baseline comparison for the more expensive **MonoT5 Reranker**.

## Experiments

We conducted three distinct fusion experiments on the `CLAPNQ` domain.

### 1. Strategy Fusion

- **Goal:** Combine different query formulations for the same search engine (ELSER).
- **Inputs:** `lastturn`, `rewrite`, `questions`.
- **Hypothesis:** Mixing query variations improves recall.

### 2. System Fusion (Ensemble)

- **Goal:** Combine different retrieval models for the same query (`rewrite`).
- **Inputs:**
  - **BM25** (Keyword/Sparse)
  - **BGE** (Semantic/Dense)
  - **ELSER** (Learned Sparse)
- **Hypothesis:** Combining orthogonal retrieval methods covers blind spots.

### 3. Weighted System Fusion

- **Goal:** Mitigate the "noise" from weaker systems by assigning higher weights to the strongest system.
- **Weights:** ELSER (3.0), BGE (1.0), BM25 (0.5).

---

## Results (CLAPNQ Domain)

| Method                  | Type            | Recall@10  | nDCG@10    | vs. Baseline |
| :---------------------- | :-------------- | :--------- | :--------- | :----------- |
| **ELSER (`rewrite`)**   | **Baseline**    | **0.7005** | **0.5780** | --           |
| **RRF Strategy Fusion** | Fusion          | 0.6483     | 0.5458     | -5.2%        |
| **RRF System Fusion**   | Fusion          | 0.6916     | 0.5608     | -0.9%        |
| **Weighted RRF**        | Fusion          | 0.6916     | 0.5608     | -0.9%        |
| **MonoT5 Reranker**     | **Reranking**   | **0.7041** | **0.5846** | **+0.4%**    |
| **Pool Oracle**         | **Upper Bound** | **0.7633** | --         | +6.3%        |

**Note:** Weighted RRF yielded identical results to Unweighted RRF because the high weight (3.0) caused ELSER's ranking to mathematically dominate the fusion score, effectively reproducing the ELSER list.

---

## Analysis

### Why RRF Failed to Beat the Baseline

1.  **The "Weak Link" Problem:** In this benchmark, the **ELSER + `rewrite`** combination is significantly stronger than other strategies (e.g., `lastturn`) and systems (e.g., BM25 R@10 = 0.38).
2.  **Dilution:** RRF treats all inputs as "votes." When you fuse a strong list with a noisy list, the noise drags down the average quality. RRF lacks the ability to distinguish between a "good rank 1" (from ELSER) and a "bad rank 1" (from BM25).

### Why MonoT5 Reranker Succeeded

- **Content-Aware Filtering:** Unlike RRF, MonoT5 reads the document text. It effectively acts as a **dynamic selector**, identifying the "Gold Nuggets" (unique relevant docs) found by weaker systems while rejecting their false positives.
- **Efficiency:** The **Pool Oracle** (0.7633) shows that the combined pool of documents _does_ contain significantly higher recall than any single system. MonoT5 captures **92%** of this potential (0.7041 / 0.7633), whereas RRF captures only 90% (and fails to rank them correctly).

### Analysis for GOVT Domain

We also analyzed the **Pool Oracle** for the `GOVT` domain, where query drift is more common.

- **ELSER Baseline:** 0.6510
- **Pool Oracle:** **0.7408** (+9.0%)
- **Implication:** The potential for fusion is even higher in `GOVT`. While RRF failed (0.6366), a content-aware reranker (MonoT5) would likely achieve significant gains by capturing this 9% gap.

### Pool Oracle Across All Domains

The theoretical upper bound for fusion (picking the best documents from BM25+BGE+ELSER) varies by domain:

| Domain     | Pool Oracle R@10 | vs. ELSER Baseline | Potential |
| :--------- | :--------------- | :----------------- | :-------- |
| **CLAPNQ** | 0.7633           | +6.3%              | High      |
| **GOVT**   | 0.7408           | +9.0%              | Very High |
| **CLOUD**  | 0.6194           | _(Baseline TBD)_   | Moderate  |
| **FIQA**   | 0.6249           | _(Baseline TBD)_   | Moderate  |

## Conclusion

**Simple mathematical fusion (RRF) is insufficient for this task.**
The performance gap between the best system (ELSER) and the others is too large for unsupervised fusion to handle. To leverage the diversity of the ensemble (the 76% Oracle potential), a **content-aware discriminator** (like MonoT5) is required to filter the pool.
