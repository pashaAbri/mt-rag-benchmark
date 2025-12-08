# MMR Clustering vs BGE Baselines Comparison

## Overview

Comparison of **MMR Clustering (k=10, λ=0.7)** approach against BGE baseline strategies:

- **Last Turn**: Current question only (no context)
- **Query Rewrite**: Human-rewritten standalone queries
- **Questions**: All questions from conversation concatenated

## Overall Performance (All Turns)

| Method                       | Recall@10  | NDCG@10    | Count |
| ---------------------------- | ---------- | ---------- | ----- |
| **Last Turn**                | 0.4277     | 0.3445     | 777   |
| **Query Rewrite**            | 0.4980     | 0.4037     | 777   |
| **Questions**                | 0.2829     | 0.2116     | 777   |
| **MMR Cluster (k10, λ=0.7)** | **0.4291** | **0.3434** | 842   |

### Key Findings

1. **MMR Clustering performs similarly to Last Turn baseline**

   - Recall@10: 0.4291 vs 0.4277 (+0.0014, +0.3%)
   - NDCG@10: 0.3434 vs 0.3445 (-0.0011, -0.3%)

2. **MMR Clustering underperforms Query Rewrite baseline**

   - Recall@10: 0.4291 vs 0.4980 (-0.0689, -13.8%)
   - NDCG@10: 0.3434 vs 0.4037 (-0.0603, -14.9%)

3. **MMR Clustering significantly outperforms Questions baseline**
   - Recall@10: 0.4291 vs 0.2829 (+0.1462, +51.7%)
   - NDCG@10: 0.3434 vs 0.2116 (+0.1318, +62.3%)

## Turn-by-Turn Performance vs Query Rewrite Baseline

| Turn   | MMR Recall@10 | MMR NDCG@10 | vs Rewrite ΔR | vs Rewrite ΔN | Count |
| ------ | ------------- | ----------- | ------------- | ------------- | ----- |
| **1**  | **0.6039**    | **0.4978**  | **+0.1059**   | **+0.0941**   | 110   |
| **2**  | 0.4690        | 0.4002      | -0.0290       | -0.0035       | 110   |
| **3**  | 0.4083        | 0.3147      | -0.0897       | -0.0890       | 110   |
| **4**  | 0.3938        | 0.3030      | -0.1042       | -0.1007       | 109   |
| **5**  | 0.3781        | 0.3077      | -0.1199       | -0.0960       | 106   |
| **6**  | 0.4158        | 0.3095      | -0.0822       | -0.0942       | 100   |
| **7**  | 0.3529        | 0.2905      | -0.1451       | -0.1132       | 89    |
| **8**  | 0.3475        | 0.2765      | -0.1505       | -0.1272       | 68    |
| **9**  | 0.3789        | 0.2652      | -0.1191       | -0.1385       | 33    |
| **10** | 0.9000        | 0.7837      | +0.4020       | +0.3800       | 5     |
| **11** | 0.7500        | 0.8319      | +0.2520       | +0.4282       | 1     |
| **12** | 1.0000        | 1.0000      | +0.5020       | +0.5963       | 1     |

### Turn-by-Turn Analysis

**Turn 1 (No Context):**

- **Best performing turn** - Outperforms Query Rewrite by +10.6% Recall@10
- This is expected since Turn 1 has no history, so MMR clustering has no context to work with
- The performance matches the "Last Turn" baseline behavior

**Turns 2-9 (With Context):**

- **Performance degrades** compared to Query Rewrite baseline
- Worst at Turn 8: -15.1% Recall@10, -12.7% NDCG@10
- This aligns with our previous analysis showing the "context shock" problem
- The MMR clustering approach struggles to effectively use conversation history

**Turns 10-12 (Very Long Conversations):**

- **Outperforms baseline** but with very small sample sizes (1-5 queries)
- These are outliers and not statistically significant

## Key Insights

### 1. **Turn 1 Advantage**

MMR Clustering performs best at Turn 1 because:

- No context to process (no clustering needed)
- Direct query matching works well
- Outperforms Query Rewrite baseline (+10.6%)

### 2. **Context Integration Challenge**

Turns 2-9 show consistent underperformance:

- The MMR clustering approach adds complexity without clear benefit
- Context from previous turns may be introducing noise
- The "context shock" problem identified in our decline analysis

### 3. **Comparison to Baselines**

- **Better than Questions**: MMR clustering is much better than concatenating all questions
- **Similar to Last Turn**: MMR clustering performs similarly to using no context
- **Worse than Query Rewrite**: Human-rewritten queries still outperform automated clustering approach

## Recommendations

1. **Turn 1**: MMR clustering works well (or use simple Last Turn approach)
2. **Turns 2+**: Consider alternative context integration strategies
3. **Topic Change Detection**: Add mechanism to detect when context becomes irrelevant
4. **Query Relevance Filtering**: Filter context by relevance to current query, not just diversity

## Conclusion

The MMR clustering approach shows promise at Turn 1 but struggles with context integration in later turns. The overall performance is similar to the Last Turn baseline, suggesting that the added complexity of clustering and MMR selection may not be providing sufficient benefit compared to simpler approaches.

The significant gap compared to Query Rewrite baseline (human-rewritten queries) indicates there's still room for improvement in automated query rewriting strategies.
