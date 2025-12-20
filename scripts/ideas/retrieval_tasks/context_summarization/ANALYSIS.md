# Context Summarization Analysis

## Executive Summary

The SELF-multi-RAG summarization technique **does not improve BM25 retrieval** on the MT-RAG benchmark. The approach actually hurts performance compared to simpler baselines.

## Key Findings

### 1. Performance Comparison

| Method                        | R@5       | R@10      | nDCG@5    | nDCG@10   |
| ----------------------------- | --------- | --------- | --------- | --------- |
| **Context Summary (ours)**    | **0.213** | **0.291** | **0.181** | **0.213** |
| BM25 Last Turn (baseline)     | 0.20      | 0.27      | 0.18      | 0.21      |
| BM25 Query Rewrite (baseline) | 0.25      | 0.33      | 0.22      | 0.25      |

Our approach is:

- **+6.5% vs Last Turn** at R@5 (marginal improvement)
- **-14.8% vs Query Rewrite** at R@5 (significant degradation)

### 2. Root Cause: Query Length vs BM25

| Method              | Avg Query Length (words) |
| ------------------- | ------------------------ |
| Last Turn           | ~10                      |
| Query Rewrite       | ~11                      |
| **Context Summary** | **~59**                  |

**BM25 is a bag-of-words model.** Longer queries with more terms:

- Dilute TF-IDF weights of important terms
- Add noisy/irrelevant matching terms
- Match more documents, reducing precision

### 3. Performance Degrades with Turn Number

| Turn | Count | Avg R@5 | Zero Recall % |
| ---- | ----- | ------- | ------------- |
| 1    | 100   | 0.385   | 44.0%         |
| 2    | 104   | 0.345   | 42.3%         |
| 3    | 102   | 0.214   | 61.8%         |
| 4    | 97    | 0.163   | 74.2%         |
| 5+   | 360   | 0.130   | 75.6%         |

Later turns have more conversation history → longer summaries → worse BM25 performance.

### 4. Summary Length Hurts Performance

| Summary Word Count     | Count | Avg R@5   |
| ---------------------- | ----- | --------- |
| 0 (Turn 1, no summary) | 100   | **0.385** |
| 1-39 (Short)           | 53    | 0.318     |
| 40-55 (Target)         | 487   | 0.186     |
| 56+ (Long)             | 123   | 0.140     |

The paper's target of 40-50 words actually performs **51% worse** than no summary.

### 5. Extra Terms Problem

446 of 763 summarized queries (58%) contain 10+ terms not in the original question.

**Example:**

- Original: "human guinea pig"
- Summary: "Guinea pigs were domesticated around 5000 BC in the Andes for meat consumption and remain a culinary staple in Peru, Bolivia, and Colombia..."
- Extra terms: cecotropes, culinary, Andes, domesticated, vegetation, etc.

These extra terms match irrelevant documents instead of the actual answer about "human guinea pig" experiments.

### 6. Per-Domain Analysis

| Domain | Total | Zero Recall % | Avg R@5   |
| ------ | ----- | ------------- | --------- |
| ClapNQ | 202   | 58.4%         | 0.213     |
| FiQA   | 180   | 76.7%         | 0.113     |
| Govt   | 193   | 54.9%         | **0.309** |
| Cloud  | 188   | 68.1%         | 0.213     |

**FiQA performs worst** (finance domain) - financial terms are very specific and the summary introduces non-financial context that hurts matching.

**Govt performs best** - government documents may have more overlap with general summarization vocabulary.

## Why SELF-multi-RAG Paper Showed Improvements

The paper tested with **Contriever** (dense retriever), not BM25:

- Dense retrievers use semantic similarity
- Longer context helps capture meaning
- Extra terms don't hurt because embeddings are semantic

For BM25:

- Lexical matching is exact
- More terms = more noise
- Focused queries work better

## Recommendations

### For BM25 Retrieval

1. **Don't use full summarization** - use shorter query rewrites
2. Keep queries under 20 words
3. Focus on key entities and question terms only

### For Dense Retrieval (BGE/ELSER)

The summarization approach may still help - should test with dense retrievers.

### Alternative Approach: Hybrid

1. Use summary for understanding/context
2. Extract key terms for BM25 query

---

## EXPERIMENT: Multi-Query Fusion + MonoT5 Reranking

### Hypothesis

Each query representation captures different signals:
- **Last turn**: Direct user intent, minimal noise
- **Query rewrite**: Contextually enriched, balanced
- **Context summary**: Rich context, potentially noisy for lexical but good for reranking

By combining BM25 retrieval from all three and reranking with MonoT5, we achieve the best of all worlds.

### Results

| Method                    | R@5    | R@10   | nDCG@5  | nDCG@10 |
| ------------------------- | ------ | ------ | ------- | ------- |
| **Fusion + MonoT5**       | **0.370** | **0.438** | **0.350** | **0.371** |
| BM25 rewrite (baseline)   | 0.247  | 0.345  | 0.218   | 0.246   |
| BM25 lastturn (baseline)  | 0.213  | 0.296  | 0.192   | 0.216   |
| Context Summary (BM25)    | 0.212  | 0.289  | 0.177   | 0.200   |

### Improvement vs Best Baseline (BM25 rewrite)

| Metric    | Improvement |
| --------- | ----------- |
| Recall@5  | **+49.6%**  |
| Recall@10 | **+26.9%**  |
| nDCG@5    | **+60.9%**  |
| nDCG@10   | **+50.6%**  |

### Per-Domain Results

| Domain | R@5    | R@10   | nDCG@5  | nDCG@10 |
| ------ | ------ | ------ | ------- | ------- |
| ClapNQ | 0.408  | 0.488  | 0.409   | 0.433   |
| Cloud  | 0.347  | 0.404  | 0.316   | 0.333   |
| FiQA   | 0.245  | 0.295  | 0.239   | 0.253   |
| Govt   | 0.480  | 0.566  | 0.436   | 0.463   |

### Why It Works

1. **Diverse Retrieval = Higher Recall**: Each query type retrieves different relevant documents
   - Avg ~21 unique docs per query from 3 sources combined
   
2. **MonoT5 Semantic Reranking**: Neural reranker understands meaning, not just keywords
   - Uses rewrite query for scoring (most informative)
   - Filters noise introduced by context summary
   - Promotes truly relevant documents

3. **Complementary Signals**:
   - Context summary may retrieve docs that BM25 rewrite misses (via additional context)
   - Last turn captures exact user intent without drift
   - Rewrite balances both

### Conclusion

**The hypothesis is validated!** Multi-query fusion + neural reranking dramatically outperforms single-query BM25 retrieval. The context summarization approach, while hurting pure BM25 retrieval, provides valuable signal when combined with other strategies and neural reranking.
3. Combine: "key_entity + reformulated_question" (~20 words max)

## Conclusion

The SELF-multi-RAG summarization technique is designed for **dense retrievers**, not lexical BM25. For BM25, the traditional query rewriting approach (short, focused) outperforms longer summarization.

To validate if summarization helps with dense retrieval, we should run with BGE or ELSER retrievers.
