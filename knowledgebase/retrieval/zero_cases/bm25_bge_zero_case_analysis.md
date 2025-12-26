# Two-Stage BM25+BGE: Zero-Case Recovery Analysis

## Overview

This document analyzes whether the two-stage BM25@500+BGE retrieval approach can recover any of the 98 zero-score cases identified in the baseline analysis.

**Hypothesis**: BM25's lexical matching with a larger candidate pool (500) combined with BGE's semantic reranking might surface relevant documents that single-stage retrievers miss.

---

## Method

| Parameter          | Value                   |
| :----------------- | :---------------------- |
| Stage 1            | BM25 (PyTerrier, local) |
| Stage 1 Candidates | Top 500                 |
| Stage 2            | BGE reranking (local)   |
| Final Top-k        | 10                      |
| Query Type         | Standard Rewrite        |
| Zero Cases Tested  | 98 (all baseline zero-score cases) |

### Pipeline

```
Query → BM25 retrieves top 500 → BGE reranks → Final top 10
```

---

## Results

| Metric                   | Value               |
| :----------------------- | :------------------ |
| **Zero Cases Recovered** | **13 / 98 (13.3%)** |
| **Still Zero**           | 85 / 98 (86.7%)     |

### Recovered Cases

| Domain | Task ID                                 | R@10  | nDCG@10 | Query                                                          |
| :----- | :-------------------------------------- | :---- | :------ | :------------------------------------------------------------- |
| ClapNQ | `dd82f0f978316e73618cf0addd369cd8<::>9` | 1.000 | 0.387   | "I like ice hockey as my sons were great players..."           |
| FiQA   | `c5518952b78b171de2d5b9317103ba62<::>7` | 1.000 | 0.315   | "can you let me know advantages of using two bank accounts?"   |
| Govt   | `ccdfb6b6f98c55047ae81b705104dbd6<::>1` | 1.000 | 0.631   | "What are the scientific objectives of NASA's Deep Impact..."  |
| ClapNQ | `694e275f1a01ad0e8ac448ad809f7930<::>6` | 0.500 | 0.264   | "I can see that play any sport soccer any other needs..."      |
| ClapNQ | `fd99b316e5e64f19ff938598aea9b285<::>4` | 0.500 | 0.307   | "Which is the most popular?"                                   |
| Cloud  | `d5b1e735a040853ed361a3dfde1b8ef0<::>1` | 0.500 | 0.204   | "does IBM offer document databases?"                           |
| Cloud  | `1dcbbeb35d4d25ba1ffb787a9f2080e2<::>1` | 0.500 | 0.177   | "How do I find specific conversations?"                        |
| Govt   | `e52ab8d5f61ccdfc3712a2608d8c2aba<::>8` | 0.500 | 0.264   | "our environment is getting worse...scary!"                    |
| ClapNQ | `33431330abb38298cc79b96b2f4fde2a<::>4` | 0.333 | 0.469   | "Why it fall?"                                                 |
| Cloud  | `f5a8ca2f2bc12180940167fb920bb018<::>5` | 0.333 | 0.148   | "Are dialogue skills necessary?"                               |
| FiQA   | `f05ba9633e1b377f9c4d64afd3da3c45<::>8` | 0.333 | 0.156   | "What is a better investment?"                                 |
| Govt   | `2cc753bcef23767c18aedad06c4405c4<::>6` | 0.333 | 0.235   | "what is electronic waste and hazardous waste here?"           |
| FiQA   | `c5518952b78b171de2d5b9317103ba62<::>3` | 0.250 | 0.113   | "if it is free, then I should make a second account..."        |

### Recovery by Domain

| Domain | Recovered | Total Zero Cases | Rate  |
| :----- | :-------- | :--------------- | :---- |
| ClapNQ | 4         | 22               | 18.2% |
| Cloud  | 3         | 28               | 10.7% |
| FiQA   | 3         | 25               | 12.0% |
| Govt   | 3         | 23               | 13.0% |

---

## Comparison with Other Approaches

| Approach                      | Zero Cases Recovered | Recovery Rate |
| :---------------------------- | :------------------- | :------------ |
| Baseline ELSER (single-stage) | 0                    | 0.0%          |
| BM25@500+ELSER                | 3                    | 3.1%          |
| **BM25@500+BGE**              | **13**               | **13.3%**     |

---
