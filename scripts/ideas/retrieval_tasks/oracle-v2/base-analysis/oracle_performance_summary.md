# Oracle Performance Analysis Summary

This document summarizes the oracle performance calculations for three retrieval strategies (lastturn, rewrite, questions) across three retrievers (BM25, ELSER, BGE) and all domains.

## Methodology

The oracle selects the best strategy per task based on nDCG@10, then reports aggregate performance across all metrics. This represents the theoretical upper bound if we could perfectly select the best strategy for each query.

## Key Findings

### Overall Oracle Improvements (All Domains)

| Retriever | Best Static Strategy | Oracle nDCG@10 | Improvement | Oracle R@10 | Improvement |
|-----------|---------------------|----------------|-------------|-------------|-------------|
| **BM25**  | Rewrite             | 0.3335         | +26.50%     | 0.4188      | +20.01%     |
| **ELSER** | Rewrite             | 0.5805         | +17.22%     | 0.6928      | +13.99%     |
| **BGE**   | Rewrite             | 0.4629         | +14.64%     | 0.5664      | +13.72%     |

**Key Insight**: BM25 shows the highest relative improvement potential (26.50% on nDCG@10), while ELSER achieves the highest absolute performance (0.5805 nDCG@10).

### Oracle Strategy Distribution (All Domains)

| Retriever | Lastturn | Rewrite | Questions |
|-----------|----------|---------|-----------|
| **BM25**  | 69.8%    | 16.7%   | 13.5%     |
| **ELSER** | 63.8%    | 23.9%   | 12.2%     |
| **BGE**   | 62.8%    | 27.4%   | 9.8%      |

**Key Insight**: Lastturn is selected most frequently across all retrievers (~63-70%), suggesting many queries don't benefit from conversation history.

---

## Detailed Results by Domain

### BM25 Results

#### ClapNQ
- **Best Static**: Rewrite (nDCG@10: 0.2937)
- **Oracle**: nDCG@10: 0.3596 (+22.46%), R@10: 0.4587 (+17.91%)
- **Distribution**: Lastturn 64.4%, Rewrite 18.8%, Questions 16.8%

#### Cloud
- **Best Static**: Rewrite (nDCG@10: 0.2329)
- **Oracle**: nDCG@10: 0.3129 (+34.37%), R@10: 0.3908 (+24.42%)
- **Distribution**: Lastturn 78.2%, Rewrite 10.1%, Questions 11.7%

#### FiQA
- **Best Static**: Rewrite (nDCG@10: 0.1737)
- **Oracle**: nDCG@10: 0.2157 (+24.17%), R@10: 0.2847 (+17.64%)
- **Distribution**: Lastturn 71.1%, Rewrite 18.9%, Questions 10.0%

#### Govt
- **Best Static**: Rewrite (nDCG@10: 0.3418)
- **Oracle**: nDCG@10: 0.4311 (+26.13%), R@10: 0.5239 (+20.17%)
- **Distribution**: Lastturn 66.2%, Rewrite 18.9%, Questions 14.9%

#### All Domains
- **Best Static**: Rewrite (nDCG@10: 0.2636)
- **Oracle**: nDCG@10: 0.3335 (+26.50%), R@10: 0.4188 (+20.01%)
- **Distribution**: Lastturn 69.8%, Rewrite 16.7%, Questions 13.5%

---

### ELSER Results

#### ClapNQ
- **Best Static**: Rewrite (nDCG@10: 0.5780)
- **Oracle**: nDCG@10: 0.6495 (+12.37%), R@10: 0.7663 (+9.39%)
- **Distribution**: Lastturn 62.0%, Rewrite 26.0%, Questions 12.0%

#### Cloud
- **Best Static**: Rewrite (nDCG@10: 0.4377)
- **Oracle**: nDCG@10: 0.5196 (+18.71%), R@10: 0.6184 (+17.11%)
- **Distribution**: Lastturn 69.7%, Rewrite 17.6%, Questions 12.8%

#### FiQA
- **Best Static**: Rewrite (nDCG@10: 0.4355)
- **Oracle**: nDCG@10: 0.5062 (+16.23%), R@10: 0.6046 (+12.85%)
- **Distribution**: Lastturn 65.0%, Rewrite 27.8%, Questions 7.2%

#### Govt
- **Best Static**: Rewrite (nDCG@10: 0.5169)
- **Oracle**: nDCG@10: 0.6327 (+22.39%), R@10: 0.7655 (+17.59%)
- **Distribution**: Lastturn 59.2%, Rewrite 24.4%, Questions 16.4%

#### All Domains
- **Best Static**: Rewrite (nDCG@10: 0.4953)
- **Oracle**: nDCG@10: 0.5805 (+17.22%), R@10: 0.6928 (+13.99%)
- **Distribution**: Lastturn 63.8%, Rewrite 23.9%, Questions 12.2%

---

### BGE Results

#### ClapNQ
- **Best Static**: Rewrite (nDCG@10: 0.4982)
- **Oracle**: nDCG@10: 0.5479 (+9.97%), R@10: 0.6770 (+11.67%)
- **Distribution**: Lastturn 56.7%, Rewrite 33.2%, Questions 10.1%

#### Cloud
- **Best Static**: Rewrite (nDCG@10: 0.3420)
- **Oracle**: nDCG@10: 0.4051 (+18.46%), R@10: 0.4904 (+16.05%)
- **Distribution**: Lastturn 71.3%, Rewrite 18.6%, Questions 10.1%

#### FiQA
- **Best Static**: Rewrite (nDCG@10: 0.3410)
- **Oracle**: nDCG@10: 0.3868 (+13.42%), R@10: 0.4784 (+14.36%)
- **Distribution**: Lastturn 64.4%, Rewrite 28.9%, Questions 6.7%

#### Govt
- **Best Static**: Rewrite (nDCG@10: 0.4199)
- **Oracle**: nDCG@10: 0.4970 (+18.37%), R@10: 0.6017 (+13.96%)
- **Distribution**: Lastturn 59.7%, Rewrite 28.4%, Questions 11.9%

#### All Domains
- **Best Static**: Rewrite (nDCG@10: 0.4037)
- **Oracle**: nDCG@10: 0.4629 (+14.64%), R@10: 0.5664 (+13.72%)
- **Distribution**: Lastturn 62.8%, Rewrite 27.4%, Questions 9.8%

---

## Cross-Retriever Comparison

### Oracle Performance by Retriever (All Domains)

| Metric    | BM25 Oracle | ELSER Oracle | BGE Oracle | Best Retriever |
|-----------|------------|--------------|------------|----------------|
| R@1       | 0.1265     | 0.2317       | 0.1771     | ELSER          |
| R@3       | 0.2424     | 0.4476       | 0.3508     | ELSER          |
| R@5       | 0.3133     | 0.5556       | 0.4378     | ELSER          |
| R@10      | 0.4188     | 0.6928       | 0.5664     | ELSER          |
| nDCG@1    | 0.2947     | 0.5238       | 0.4118     | ELSER          |
| nDCG@3    | 0.2624     | 0.4825       | 0.3763     | ELSER          |
| nDCG@5    | 0.2894     | 0.5209       | 0.4082     | ELSER          |
| nDCG@10   | 0.3335     | 0.5805       | 0.4629     | ELSER          |

**Key Insight**: ELSER achieves the highest oracle performance across all metrics, with BGE second and BM25 third.

### Oracle Improvement Potential by Retriever

| Retriever | Avg Improvement (nDCG@10) | Avg Improvement (R@10) |
|-----------|---------------------------|------------------------|
| **BM25**  | +26.50%                   | +20.01%                |
| **ELSER** | +17.22%                   | +13.99%                |
| **BGE**   | +14.64%                   | +13.72%                |

**Key Insight**: While BM25 has the highest relative improvement potential, ELSER achieves the highest absolute performance even with oracle selection.

---

## Domain-Specific Patterns

### Highest Oracle Improvement by Domain

| Domain | Retriever | Improvement (nDCG@10) |
|--------|-----------|----------------------|
| Cloud  | BM25      | +34.37%              |
| Cloud  | ELSER     | +18.71%              |
| Cloud  | BGE       | +18.46%              |
| Govt   | ELSER     | +22.39%              |
| ClapNQ | BM25      | +22.46%              |

**Key Insight**: Cloud domain shows the highest improvement potential across retrievers, suggesting more variability in which strategy works best.

### Most Consistent Strategy Selection

- **ClapNQ**: BGE shows most balanced distribution (56.7% Lastturn, 33.2% Rewrite)
- **Cloud**: BM25 shows strongest Lastturn preference (78.2%)
- **FiQA**: BGE shows strongest Rewrite preference (28.9%)
- **Govt**: Most balanced across retrievers (~60% Lastturn, ~25% Rewrite)

---

## Implications

1. **Strategy Selection Value**: Oracle improvements range from 14-26%, indicating significant value in developing query routing/classification systems.

2. **Retriever Performance**: ELSER consistently outperforms BGE and BM25, even with oracle selection, suggesting it's the best base retriever.

3. **Strategy Preferences**: Lastturn is selected most frequently (~63-70%), suggesting many queries don't benefit from conversation history. However, Rewrite is selected more often with better retrievers (ELSER, BGE).

4. **Domain Variability**: Cloud domain shows highest improvement potential, suggesting more complex query patterns that benefit from strategy selection.

5. **Questions Strategy**: Consistently underperforms, selected only 7-17% of the time, suggesting concatenating all questions is rarely optimal.

---

## Files Generated

All detailed results are saved as JSON files:
- `oracle_bm25_{domain}_results.json`
- `oracle_elser_{domain}_results.json`
- `oracle_bge_{domain}_results.json`

Where `{domain}` is one of: `clapnq`, `cloud`, `fiqa`, `govt`, `all`

