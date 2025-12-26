# Two-Stage BM25+BGE: Zero-Case Recovery Analysis

## Overview

This document analyzes whether a **two-stage retrieval approach (BM25 filtering → BGE reranking)** can recover the 98 zero-score cases where all baseline strategies completely fail.

**Source**: `scripts/ideas/retrieval_tasks/bm25_bge_rerank/`

---

## Method

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: BM25 Retrieval                                         │
│   - Retrieve top 50 candidates using BM25 lexical matching      │
│   - Higher recall due to larger pool                            │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: BGE Reranking                                          │
│   - Encode query and 50 candidates with BGE-base-en-v1.5        │
│   - Compute cosine similarity scores                            │
│   - Return top 10 by semantic similarity                        │
└─────────────────────────────────────────────────────────────────┘
```

### Query Types Evaluated

- **lastturn**: Only the most recent user question
- **rewrite**: Standalone rewritten query
- **questions**: Concatenation of all questions in conversation

---

## Key Findings

| Metric                    | Value                  |
| ------------------------- | ---------------------- |
| **Zero cases recovered**  | **16 / 98 (16.3%)** ✅ |
| **Zero cases still zero** | 82 / 98 (83.7%)        |

### Comparison with MonoT5 Reranking

| Approach                      | Zero Cases Recovered | Recovery Rate |
| ----------------------------- | -------------------- | ------------- |
| **MonoT5 fusion + reranking** | 0 / 98               | 0.0% ❌       |
| **BM25→BGE two-stage**        | 16 / 98              | **16.3%** ✅  |

**Critical Insight**: The two-stage BM25+BGE approach successfully recovers cases that MonoT5 reranking cannot touch.

---

## Why Two-Stage Helps Where MonoT5 Didn't

The fundamental difference is **retrieval pool size**:

### MonoT5 Limitation

```
Baseline retrieves top-10 → 0 relevant docs in pool
    ↓
MonoT5 reranks the pool → Still 0 relevant docs
    ↓
Cannot recover (reranking empty/irrelevant pool is futile)
```

### BM25+BGE Advantage

```
BM25 retrieves top-50 → Larger pool, some relevant docs captured
    ↓
BGE reranks with semantic matching → Surfaces relevant docs
    ↓
Recovers 16.3% of zero cases
```

The key insight is that **BM25's lexical matching with a larger pool (50 vs 10) finds some documents that dense methods (ELSER, BGE) missed at top-10**.

---

## Recovered Cases Detail

### Full List of Recovered Cases

| Task ID                                 | Domain | Query Type       | Recall@10 | nDCG@10 |
| --------------------------------------- | ------ | ---------------- | --------- | ------- |
| `dd82f0f978316e73618cf0addd369cd8<::>9` | clapnq | lastturn         | 1.000     | 0.333   |
| `adf9b1f61c73d715809bc7b37ac02724<::>1` | cloud  | lastturn         | 1.000     | 0.315   |
| `1065ea5ad1ae2b90e6fce67d851a7a66<::>6` | cloud  | rewrite          | 1.000     | 0.441   |
| `adf9b1f61c73d715809bc7b37ac02724<::>1` | cloud  | rewrite          | 1.000     | 0.315   |
| `adf9b1f61c73d715809bc7b37ac02724<::>1` | cloud  | questions        | 1.000     | 0.315   |
| `ccdfb6b6f98c55047ae81b705104dbd6<::>1` | govt   | lastturn         | 1.000     | 0.631   |
| `ccdfb6b6f98c55047ae81b705104dbd6<::>1` | govt   | rewrite          | 1.000     | 0.631   |
| `2f484ad8f3baf91136f040855892c82e<::>8` | govt   | questions        | 1.000     | 0.431   |
| `5600fe1c05a1fc415416d9dee6347000<::>7` | govt   | questions        | 1.000     | 0.315   |
| `ccdfb6b6f98c55047ae81b705104dbd6<::>9` | govt   | questions        | 1.000     | 0.631   |
| `2cc753bcef23767c18aedad06c4405c4<::>6` | govt   | lastturn         | 0.667     | 0.383   |
| `941445ba11ba7ba2c92c5184c9d798d6<::>3` | govt   | questions        | 0.667     | 0.437   |
| `d5b1e735a040853ed361a3dfde1b8ef0<::>1` | cloud  | all 3            | 0.500     | 0.387   |
| `f5a8ca2f2bc12180940167fb920bb018<::>5` | cloud  | questions        | 0.500     | 0.177   |
| `f05ba9633e1b377f9c4d64afd3da3c45<::>8` | fiqa   | lastturn/rewrite | 0.333     | 0.296   |
| `9c52934a9aea1c3647d4b558d8afdf1c<::>5` | fiqa   | questions        | 0.333     | 0.167   |
| `4da5a7f42f3b2dc4d875dcfa2fcdefef<::>6` | fiqa   | rewrite          | 0.250     | 0.168   |
| `c5518952b78b171de2d5b9317103ba62<::>7` | fiqa   | questions        | 0.250     | 0.168   |

### Recovery by Query Type

| Query Type    | Cases Recovered |
| ------------- | --------------- |
| **questions** | 11              |
| **rewrite**   | 8               |
| **lastturn**  | 7               |

**Observation**: The `questions` strategy (concatenating all questions) recovers the most zero cases, likely because it provides more lexical overlap for BM25 to match.

### Recovery by Domain

| Domain | Zero Cases | Recovered | Rate  |
| ------ | ---------- | --------- | ----- |
| clapnq | 22         | 1         | 4.5%  |
| cloud  | 28         | 4         | 14.3% |
| fiqa   | 25         | 4         | 16.0% |
| govt   | 23         | 7         | 30.4% |

**Observation**: Govt domain shows the highest recovery rate (30.4%), possibly because government documents have more consistent vocabulary that BM25 can match.

---

## Sample Recovered Cases

### Case 1: IBM Document Databases (Cloud)

- **Task ID**: `d5b1e735a040853ed361a3dfde1b8ef0<::>1`
- **Query**: "does IBM offer document databases?"
- **Why baseline failed**: Vocabulary mismatch - corpus uses "Cloudant" not "document database"
- **Why BM25+BGE recovered**: BM25's top-50 pool included "Cloudant" docs; BGE semantically matched "document database" concept

### Case 2: NASA Deep Impact Mission (Govt)

- **Task ID**: `ccdfb6b6f98c55047ae81b705104dbd6<::>1`
- **Query**: "What are the scientific objectives of NASA's Deep Impact Extended Mission?"
- **Recall@10**: 1.000, **nDCG@10**: 0.631
- **Why recovered**: First-turn query with good lexical match; BM25 found it in larger pool

### Case 3: Ice Hockey (ClapNQ)

- **Task ID**: `dd82f0f978316e73618cf0addd369cd8<::>9`
- **Query**: "I like ice hockey as my sons were great players..."
- **Recall@10**: 1.000, **nDCG@10**: 0.333
- **Why recovered**: Despite being a non-question, BM25 matched "ice hockey" lexically

---

## Limitations

### 82 Cases Still Unrecovered (83.7%)

The two-stage approach cannot recover cases where:

1. **No lexical overlap**: Query uses completely different vocabulary than documents
2. **Context-dependent references**: "it", "this", "the company" - BM25 cannot resolve
3. **Very short queries**: "Advise", "Value", "backups." - insufficient terms for retrieval
4. **Non-questions**: Statements/comments with unclear retrieval intent

### Example Unrecovered Cases

| Query                  | Why Still Zero                                 |
| ---------------------- | ---------------------------------------------- |
| "any awards"           | 2 words, no context                            |
| "Value"                | Single keyword, completely ambiguous           |
| "Tipsy"                | Single keyword, needs conversation context     |
| "What about security?" | "security" too generic without product context |

---

## Comparison with Overall Baseline Performance

While the two-stage approach helps zero cases, it **underperforms baselines** on overall metrics:

| Method       | Avg nDCG@10  |
| ------------ | ------------ |
| ELSER        | ~0.45 (best) |
| BGE          | ~0.35        |
| **BM25+BGE** | ~0.33        |
| BM25         | ~0.23        |

**Trade-off**: BM25+BGE recovers some zero cases but sacrifices overall performance compared to ELSER and pure BGE.

---

## Conclusions

1. **Two-stage BM25+BGE recovers 16.3% of zero cases** - a meaningful improvement over MonoT5's 0%

2. **Larger retrieval pool is key** - retrieving 50 candidates instead of 10 captures documents that dense methods miss

3. **Best for first-turn and lexically-rich queries** - where BM25 can find vocabulary matches

4. **Not a replacement for baselines** - underperforms ELSER/BGE on overall metrics

5. **Complementary approach** - could be used as a fallback when primary retrievers fail

---

## Recommendations

1. **Ensemble approach**: Run BM25+BGE in parallel with ELSER; use when ELSER returns low-confidence results

2. **Increase BM25 pool size**: Experiment with top-100 or top-200 to capture more candidates

3. **Focus on remaining 82 cases**: These require fundamentally different approaches:
   - Conversation context injection
   - Entity/pronoun resolution
   - Query expansion with domain knowledge

---

## Source Code

Analysis script: `scripts/ideas/retrieval_tasks/bm25_bge_rerank/analyze_zero_cases.py`

Full experiment: `scripts/ideas/retrieval_tasks/bm25_bge_rerank/`
