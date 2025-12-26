# Two-Stage BM25+ELSER: Zero-Case Recovery Analysis

## Overview

This document analyzes whether the two-stage BM25@500+ELSER retrieval approach can recover any of the 98 zero-score cases identified in the baseline analysis.

**Hypothesis**: BM25's lexical matching might retrieve documents that ELSER's semantic matching misses, providing a larger candidate pool for ELSER to rerank.

---

## Method

| Parameter          | Value                                 |
| :----------------- | :------------------------------------ |
| Stage 1            | BM25 (PyTerrier, local)               |
| Stage 1 Candidates | Top 500                               |
| Stage 2            | ELSER reranking (Elasticsearch Cloud) |
| Final Top-k        | 10                                    |
| Query Type         | Standard Rewrite                      |
| Zero Cases Tested  | 98 (all baseline zero-score cases)    |

### Pipeline

```
Query → BM25 retrieves top 500 → ELSER reranks → Final top 10
```

---

## Results

| Metric                   | Value             |
| :----------------------- | :---------------- |
| **Zero Cases Recovered** | **3 / 98 (3.1%)** |
| **Still Zero**           | 95 / 98 (96.9%)   |

### Recovered Cases

| Domain | Task ID                                 | R@10  | Query                                                        |
| :----- | :-------------------------------------- | :---- | :----------------------------------------------------------- |
| FiQA   | `c5518952b78b171de2d5b9317103ba62<::>7` | 1.000 | "can you let me know advantages of using two bank accounts?" |
| Govt   | `5600fe1c05a1fc415416d9dee6347000<::>7` | 1.000 | "I meant for a lunar eclipse"                                |
| Govt   | `e52ab8d5f61ccdfc3712a2608d8c2aba<::>8` | 0.250 | "our environment is getting worse...scary!"                  |

### Recovery by Domain

| Domain | Recovered | Total Zero Cases | Rate |
| :----- | :-------- | :--------------- | :--- |
| ClapNQ | 0         | 22               | 0.0% |
| Cloud  | 0         | 28               | 0.0% |
| FiQA   | 1         | 25               | 4.0% |
| Govt   | 2         | 23               | 8.7% |

---

## Analysis

### Why Only 3 Cases Recovered?

1. **Context-Dependent Queries**: The 98 zero cases are primarily context-dependent follow-ups (74.5%). Neither BM25 nor ELSER can resolve implicit references like "it", "the movement", or "Clipper" without conversational context.

2. **Query Rewrite Limitations**: We are using the `rewrite` query. If the rewrite doesn't properly resolve context, expanding the candidate pool doesn't help.

### Look into the 3 Were Recovered

The 3 recovered cases share a pattern: **the BM25 stage found lexically matching documents that ELSER had ranked lower or missed entirely**.

- **FiQA case**: "two bank accounts" has clear lexical overlap with relevant documents
- **Govt lunar eclipse case**: Specific terminology that BM25 matches well
- **Govt environment case**: General vocabulary with broader lexical reach

---

## Comparison with Other Approaches

| Approach                      | Zero Cases Recovered | Recovery Rate |
| :---------------------------- | :------------------- | :------------ |
| Baseline ELSER (single-stage) | 0                    | 0.0%          |
| **BM25@500+ELSER**            | **3**                | **3.1%**      |

### Key Insight

BM25+ELSER provides a **marginal improvement** over single-stage ELSER and MonoT5 reranking. However:

- 3.1% recovery is not practically significant
- 95% of zero cases remain unsolved
- The problem is fundamentally in **query formulation**, not retrieval diversity

---

## Related Documents

- [Baseline Zero-Case Analysis](baseline_zero_case_analysis.md) — Full analysis of the 98 zero-score cases
- [BM25+ELSER Results](../two_stage_retrieval/bm25_elser_results.md) — Overall two-stage retrieval performance
