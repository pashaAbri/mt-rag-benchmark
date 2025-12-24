# Cross-Retriever Oracle Performance Analysis

This document summarizes the cross-retriever oracle performance, which selects the best combination of retriever (BM25, ELSER, BGE) and strategy (lastturn, rewrite, questions) for each task - **9 total combinations**.

## Methodology

The cross-retriever oracle selects the best retriever+strategy combination per task based on nDCG@10, then reports aggregate performance across all metrics. This represents the theoretical upper bound if we could perfectly select both the best retriever AND the best strategy for each query.

## Key Findings

### Overall Cross-Retriever Oracle Performance (All Domains)

| Metric    | Best Static (ELSER+Rewrite) | Cross-Retriever Oracle | Improvement |
|-----------|------------------------------|------------------------|-------------|
| R@1       | 0.1871                      | 0.2790                 | +49.12%     |
| R@3       | 0.3723                      | 0.5127                 | +37.73%     |
| R@5       | 0.4761                      | 0.6122                 | +28.58%     |
| R@10      | 0.6078                      | 0.7352                 | +20.95%     |
| nDCG@1    | 0.4286                      | 0.6358                 | +48.35%     |
| nDCG@3    | 0.3992                      | 0.5619                 | +40.76%     |
| nDCG@5    | 0.4378                      | 0.5932                 | +35.51%     |
| nDCG@10   | 0.4953                      | 0.6459                 | **+30.42%** |

**Key Insight**: Cross-retriever oracle achieves **+30.42% improvement** on nDCG@10 over the best static combination (ELSER+Rewrite), compared to **+17.22%** improvement from single-retriever oracle (ELSER only).

### Cross-Retriever Oracle Combination Distribution (All Domains)

| Retriever | Strategy   | Count | Percentage |
|-----------|------------|-------|------------|
| ELSER     | Lastturn   | 241   | 31.0%      |
| BM25      | Lastturn   | 146   | 18.8%      |
| ELSER     | Rewrite    | 115   | 14.8%      |
| BGE       | Lastturn   | 75    | 9.7%       |
| BGE       | Rewrite    | 56    | 7.2%       |
| ELSER     | Questions  | 49    | 6.3%       |
| BM25      | Questions  | 44    | 5.7%       |
| BM25      | Rewrite    | 30    | 3.9%       |
| BGE       | Questions  | 21    | 2.7%       |

**Key Insights**:
1. **ELSER+Lastturn** is selected most frequently (31.0%), showing that ELSER with simple queries often outperforms other combinations
2. **BM25+Lastturn** is second (18.8%), indicating BM25 can be competitive for certain queries
3. **Lastturn strategy dominates**: 59.5% of selections use Lastturn (241+146+75), confirming many queries don't need conversation history
4. **Questions strategy is rarely optimal**: Only 14.7% of selections (49+44+21)

---

## Comparison: Single-Retriever vs Cross-Retriever Oracle

### Performance Comparison (All Domains)

| Metric    | Single-Retriever Oracle (ELSER) | Cross-Retriever Oracle | Additional Gain |
|-----------|----------------------------------|------------------------|-----------------|
| R@1       | 0.2317                          | 0.2790                 | +20.4%          |
| R@3       | 0.4476                          | 0.5127                 | +14.6%          |
| R@5       | 0.5556                          | 0.6122                 | +10.2%          |
| R@10      | 0.6928                          | 0.7352                 | +6.1%           |
| nDCG@1    | 0.5238                          | 0.6358                 | +21.4%          |
| nDCG@3    | 0.4825                          | 0.5619                 | +16.5%          |
| nDCG@5    | 0.5209                          | 0.5932                 | +13.9%          |
| nDCG@10   | 0.5805                          | 0.6459                 | **+11.3%**      |

**Key Insight**: Cross-retriever oracle provides an additional **+11.3% improvement** on nDCG@10 beyond single-retriever oracle, showing significant value in retriever selection.

### Improvement Breakdown

| Approach                    | nDCG@10 | Improvement over Best Static |
|-----------------------------|---------|------------------------------|
| Best Static (ELSER+Rewrite)| 0.4953  | Baseline                     |
| Single-Retriever Oracle    | 0.5805  | +17.22%                      |
| Cross-Retriever Oracle     | 0.6459  | **+30.42%**                  |

**Key Insight**: Cross-retriever oracle achieves **+30.42% improvement**, which is **1.77x** the improvement of single-retriever oracle (+17.22%).

---

## Detailed Results by Domain

### ClapNQ

- **Best Static**: ELSER+Rewrite (nDCG@10: 0.5780)
- **Cross-Retriever Oracle**: nDCG@10: 0.7032 (+21.65%), R@10: 0.8081 (+15.35%)
- **Top Combinations**:
  - ELSER+Lastturn: 33.7%
  - ELSER+Rewrite: 15.9%
  - BM25+Lastturn: 13.9%

### Cloud

- **Best Static**: ELSER+Rewrite (nDCG@10: 0.4377)
- **Cross-Retriever Oracle**: nDCG@10: 0.6060 (+38.45%), R@10: 0.6831 (+29.37%)
- **Top Combinations**:
  - ELSER+Lastturn: 30.3%
  - BM25+Lastturn: 25.0%
  - ELSER+Rewrite: 10.6%

**Key Insight**: Cloud shows the highest improvement potential (+38.45%), with BM25+Lastturn being competitive (25% of selections).

### FiQA

- **Best Static**: ELSER+Rewrite (nDCG@10: 0.4355)
- **Cross-Retriever Oracle**: nDCG@10: 0.5628 (+29.23%), R@10: 0.6279 (+17.21%)
- **Top Combinations**:
  - ELSER+Lastturn: 35.0%
  - ELSER+Rewrite: 17.2%
  - BM25+Lastturn: 14.4%

### Govt

- **Best Static**: ELSER+Rewrite (nDCG@10: 0.5169)
- **Cross-Retriever Oracle**: nDCG@10: 0.6985 (+35.12%), R@10: 0.8044 (+23.57%)
- **Top Combinations**:
  - ELSER+Lastturn: 25.4%
  - BM25+Lastturn: 21.9%
  - ELSER+Rewrite: 15.4%

---

## Retriever Selection Patterns

### When Each Retriever is Selected (All Domains)

| Retriever | Total Selections | Percentage | Avg nDCG@10 When Selected |
|-----------|------------------|------------|---------------------------|
| **ELSER** | 405              | 52.1%      | ~0.65                     |
| **BM25**  | 220              | 28.3%      | ~0.60                     |
| **BGE**   | 152              | 19.6%      | ~0.58                     |

**Key Insight**: ELSER is selected most often (52.1%), but BM25 is competitive for 28.3% of queries, suggesting retriever selection is valuable.

### Strategy Selection Patterns (All Domains)

| Strategy  | Total Selections | Percentage |
|-----------|------------------|------------|
| **Lastturn** | 462            | 59.5%      |
| **Rewrite**  | 201            | 25.9%      |
| **Questions** | 114           | 14.7%      |

**Key Insight**: Lastturn dominates (59.5%), but Rewrite is still valuable (25.9%), especially with better retrievers.

---

## Cross-Retriever vs Single-Retriever Oracle Comparison

### Additional Value from Retriever Selection

| Domain | Single-Retriever Oracle (nDCG@10) | Cross-Retriever Oracle (nDCG@10) | Additional Gain |
|--------|-----------------------------------|----------------------------------|-----------------|
| ClapNQ | 0.6495                            | 0.7032                           | +8.3%           |
| Cloud  | 0.5196                            | 0.6060                           | +16.6%          |
| FiQA   | 0.5062                            | 0.5628                           | +11.2%          |
| Govt   | 0.6327                            | 0.6985                           | +10.4%          |
| All    | 0.5805                            | 0.6459                           | **+11.3%**      |

**Key Insight**: Cloud domain shows the highest additional value from retriever selection (+16.6%), likely due to more diverse query patterns.

---

## Implications

1. **Dual Selection Value**: Cross-retriever oracle achieves **+30.42% improvement**, significantly higher than single-retriever oracle (+17.22%), showing value in both retriever AND strategy selection.

2. **Retriever Diversity**: While ELSER is selected most (52.1%), BM25 is competitive for 28.3% of queries, indicating retriever selection is valuable.

3. **Strategy Preferences**: Lastturn dominates (59.5%), but Rewrite is valuable (25.9%), especially with ELSER and BGE.

4. **Domain Variability**: Cloud shows highest improvement potential (+38.45%), suggesting more complex patterns that benefit from dual selection.

5. **Questions Strategy**: Consistently underperforms (14.7% selection rate), confirming concatenating all questions is rarely optimal.

6. **Practical Implications**: 
   - A query routing system should consider both retriever and strategy selection
   - BM25 can be competitive for certain query types, not just ELSER
   - Lastturn is often sufficient, but Rewrite adds value for ~26% of queries

---

## Files Generated

All detailed results are saved as JSON files:
- `cross_retriever_oracle_{domain}_results.json`

Where `{domain}` is one of: `clapnq`, `cloud`, `fiqa`, `govt`, `all`

Each file contains:
- Individual combination performances (all 9 combinations)
- Oracle performance (best combination per task)
- Combination distribution (which combinations were selected)

