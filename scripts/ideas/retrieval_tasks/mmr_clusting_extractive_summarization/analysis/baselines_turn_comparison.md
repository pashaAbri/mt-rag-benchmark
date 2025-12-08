# Turn-by-Turn Analysis: Baselines vs MMR Clustering

## Overview

This analysis examines how each baseline method (Last Turn, Query Rewrite, Questions) and MMR Clustering perform across conversation turns.

## Individual Baseline Turn-by-Turn Performance

### 1. Last Turn (No Context)

**Strategy**: Use only the current question, ignoring conversation history.

| Turn | Recall@10 | NDCG@10 | Count |
| ---- | --------- | ------- | ----- |
| 1    | 0.6513    | 0.5369  | 102   |
| 2    | 0.4392    | 0.3676  | 106   |
| 3    | 0.3412    | 0.2625  | 103   |
| 4    | 0.4060    | 0.3022  | 99    |
| 5    | 0.3846    | 0.3187  | 96    |
| 6    | 0.4148    | 0.3126  | 87    |
| 7    | 0.3571    | 0.2958  | 80    |
| 8    | 0.4015    | 0.3302  | 65    |
| 9    | 0.3245    | 0.2591  | 32    |

**Summary:**

- **Overall**: Recall@10 = 0.4277, NDCG@10 = 0.3445
- **Best Turn**: 12 (Recall@10 = 1.0000) - _outlier with n=1_
- **Worst Turn**: 9 (Recall@10 = 0.3245)
- **Pattern**: Sharp drop from Turn 1 → Turn 2 (-32.6%), then relatively stable around 0.35-0.41

**Key Insight**: Without context, performance degrades significantly after Turn 1, as expected. The system cannot resolve references or understand follow-up questions.

---

### 2. Query Rewrite (Human-Rewritten)

**Strategy**: Human experts rewrite queries to be standalone, incorporating necessary context.

| Turn | Recall@10 | NDCG@10 | Count |
| ---- | --------- | ------- | ----- |
| 1    | 0.6513    | 0.5369  | 102   |
| 2    | 0.5434    | 0.4511  | 106   |
| 3    | 0.4907    | 0.3912  | 103   |
| 4    | 0.4701    | 0.3662  | 99    |
| 5    | 0.4442    | 0.3598  | 96    |
| 6    | 0.4736    | 0.3778  | 87    |
| 7    | 0.4194    | 0.3508  | 80    |
| 8    | 0.4085    | 0.3396  | 65    |
| 9    | 0.5157    | 0.3646  | 32    |

**Summary:**

- **Overall**: Recall@10 = 0.4980, NDCG@10 = 0.4038
- **Best Turn**: 12 (Recall@10 = 1.0000) - _outlier with n=1_
- **Worst Turn**: 8 (Recall@10 = 0.4085)
- **Pattern**: Gradual decline from Turn 1 → Turn 8, but maintains much higher performance than Last Turn

**Key Insight**: Human rewriting is highly effective. The decline is gradual (-16.6% from Turn 1 → Turn 2) compared to Last Turn (-32.6%), and performance stays above 0.40 for most turns.

---

### 3. Questions (All Questions Concatenated)

**Strategy**: Concatenate all questions from the conversation into a single query.

| Turn | Recall@10 | NDCG@10 | Count |
| ---- | --------- | ------- | ----- |
| 1    | 0.6513    | 0.5369  | 102   |
| 2    | 0.5063    | 0.4013  | 106   |
| 3    | 0.3071    | 0.2260  | 103   |
| 4    | 0.2304    | 0.1580  | 99    |
| 5    | 0.1450    | 0.0893  | 96    |
| 6    | 0.1295    | 0.0800  | 87    |
| 7    | 0.1056    | 0.0751  | 80    |
| 8    | 0.1487    | 0.0836  | 65    |
| 9    | 0.0208    | 0.0117  | 32    |

**Summary:**

- **Overall**: Recall@10 = 0.2829, NDCG@10 = 0.2116
- **Best Turn**: 1 (Recall@10 = 0.6513)
- **Worst Turn**: 11 (Recall@10 = 0.0000) - _outlier with n=1_
- **Pattern**: **Catastrophic decline** - drops to 0.02 by Turn 9

**Key Insight**: Concatenating all questions creates severe query drift. The longer the conversation, the worse it gets. This is the worst-performing strategy for multi-turn conversations.

---

### 4. MMR Cluster (k=10, λ=0.7)

**Strategy**: Extract sentences from history, cluster them, select diverse representatives via MMR, then rewrite query using LLM.

| Turn | Recall@10 | NDCG@10 | Count |
| ---- | --------- | ------- | ----- |
| 1    | 0.6039    | 0.4978  | 110   |
| 2    | 0.4690    | 0.4002  | 110   |
| 3    | 0.4083    | 0.3147  | 110   |
| 4    | 0.3938    | 0.3030  | 109   |
| 5    | 0.3781    | 0.3077  | 106   |
| 6    | 0.4158    | 0.3095  | 100   |
| 7    | 0.3529    | 0.2905  | 89    |
| 8    | 0.3475    | 0.2765  | 68    |
| 9    | 0.3789    | 0.2652  | 33    |

**Summary:**

- **Overall**: Recall@10 = 0.4291, NDCG@10 = 0.3434
- **Best Turn**: 12 (Recall@10 = 1.0000) - _outlier with n=1_
- **Worst Turn**: 8 (Recall@10 = 0.3475)
- **Pattern**: Sharp drop from Turn 1 → Turn 2 (-22.3%), then stabilizes around 0.35-0.42

**Key Insight**: MMR clustering performs similarly to Last Turn baseline, suggesting the clustering approach isn't providing significant benefit over simply ignoring context.

---

## Combined Comparison: All Methods Turn-by-Turn

| Turn | Last Turn | Query Rewrite | Questions | MMR Cluster |
| | R@10 N@10 | R@10 N@10 | R@10 N@10 | R@10 N@10 |
|------|------------|---------------|------------|-------------|
| **1** | 0.6513 0.5369 | **0.6513** **0.5369** | 0.6513 0.5369 | 0.6039 0.4978 |
| **2** | 0.4392 0.3676 | **0.5434** **0.4511** | 0.5063 0.4013 | 0.4690 0.4002 |
| **3** | 0.3412 0.2625 | **0.4907** **0.3912** | 0.3071 0.2260 | 0.4083 0.3147 |
| **4** | 0.4060 0.3022 | **0.4701** **0.3662** | 0.2304 0.1580 | 0.3938 0.3030 |
| **5** | 0.3846 0.3187 | **0.4442** **0.3598** | 0.1450 0.0893 | 0.3781 0.3077 |
| **6** | 0.4148 0.3126 | **0.4736** **0.3778** | 0.1295 0.0800 | 0.4158 0.3095 |
| **7** | 0.3571 0.2958 | **0.4194** **0.3508** | 0.1056 0.0751 | 0.3529 0.2905 |
| **8** | 0.4015 0.3302 | **0.4085** **0.3396** | 0.1487 0.0836 | 0.3475 0.2765 |
| **9** | 0.3245 0.2591 | **0.5157** **0.3646** | 0.0208 0.0117 | 0.3789 0.2652 |

_Note: Turns 10-12 have very small sample sizes (1-5 queries) and are outliers._

---

## Key Findings

### 1. **Turn 1 Performance (No Context)**

All methods perform identically at Turn 1 (0.65 Recall@10) except MMR Clustering (0.60), which is slightly lower. This makes sense since:

- Turn 1 has no history to process
- All methods are essentially doing the same thing: retrieving based on the original question
- MMR clustering adds overhead without benefit when there's no context

### 2. **Turn 2 Performance (First Context)**

The **"Context Shock"** is visible across all methods:

| Method            | Turn 1 → Turn 2 Decline | Performance at Turn 2 |
| ----------------- | ----------------------- | --------------------- |
| **Query Rewrite** | -16.6%                  | **0.5434** (Best)     |
| **Questions**     | -22.3%                  | 0.5063                |
| **MMR Cluster**   | -22.3%                  | 0.4690                |
| **Last Turn**     | -32.6%                  | 0.4392 (Worst)        |

**Key Insight**: Query Rewrite handles the context transition best, with only a 16.6% decline. MMR Clustering performs similarly to Questions concatenation, suggesting the clustering isn't effectively filtering noise.

### 3. **Turns 3-9 Performance (With Context)**

**Query Rewrite** maintains the best performance:

- Stays above 0.40 Recall@10 for all turns
- Gradual, controlled decline
- Best overall performance

**MMR Cluster** and **Last Turn** perform similarly:

- Both stabilize around 0.35-0.42 Recall@10
- MMR clustering doesn't provide clear advantage over ignoring context

**Questions** shows catastrophic decline:

- Drops to 0.02 by Turn 9
- Query drift makes this approach unusable for long conversations

### 4. **Performance Ranking (Turns 2-9)**

1. **Query Rewrite**: 0.4085 - 0.5434 Recall@10 (Best)
2. **MMR Cluster**: 0.3475 - 0.4690 Recall@10
3. **Last Turn**: 0.3245 - 0.4392 Recall@10
4. **Questions**: 0.0208 - 0.5063 Recall@10 (Worst for later turns)

---

## Comparative Analysis: MMR Clustering vs Baselines

### vs Last Turn Baseline

**Similar Performance:**

- Overall: 0.4291 vs 0.4277 (+0.3% Recall@10)
- Turn-by-turn: Very similar patterns, both show "context shock" at Turn 2
- **Conclusion**: MMR clustering provides minimal benefit over ignoring context entirely

### vs Query Rewrite Baseline

**Significant Underperformance:**

- Overall: 0.4291 vs 0.4980 (-13.8% Recall@10)
- Turn 2: 0.4690 vs 0.5434 (-13.7% Recall@10)
- **Conclusion**: Human rewriting is much more effective than automated clustering approach

### vs Questions Baseline

**Significant Outperformance:**

- Overall: 0.4291 vs 0.2829 (+51.7% Recall@10)
- Turn 9: 0.3789 vs 0.0208 (+1721% Recall@10)
- **Conclusion**: MMR clustering is much better than concatenating all questions, but this is a low bar

---

## Recommendations

### For Turn 1

- **Use Last Turn or Query Rewrite** - Both perform identically (0.65 Recall@10)
- MMR clustering adds unnecessary overhead

### For Turn 2+

- **Query Rewrite is the gold standard** - Human rewriting maintains best performance
- **MMR Clustering needs improvement** - Currently performs similarly to ignoring context
- **Questions concatenation should be avoided** - Catastrophic performance degradation

### Areas for MMR Clustering Improvement

1. **Better Context Selection**: Current clustering may be selecting irrelevant context
2. **Topic Change Detection**: Need to detect when context becomes irrelevant
3. **Query Relevance Filtering**: Filter context by relevance to current query, not just diversity
4. **Turn 2 Special Handling**: The first turn with context needs different treatment

---

## Conclusion

The turn-by-turn analysis reveals that:

1. **Query Rewrite (human)** is the clear winner, maintaining strong performance across all turns
2. **MMR Clustering** performs similarly to **Last Turn**, suggesting the clustering approach isn't providing sufficient benefit
3. **Questions concatenation** fails catastrophically for multi-turn conversations
4. All methods show a "context shock" at Turn 2, but Query Rewrite handles it best

The MMR clustering approach shows promise but needs refinement to match the performance of human query rewriting, particularly in how it selects and uses context from previous turns.
