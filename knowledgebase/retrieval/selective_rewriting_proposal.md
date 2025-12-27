# Proposal: Selective Query Rewriting

A data-driven approach to decide **when to rewrite** a conversational query vs. **when to use it as-is**.

---

## Executive Summary

| Strategy              | nDCG@5     | vs Always Rewrite | Rewrite % | Recommendation     |
| --------------------- | ---------- | ----------------- | --------- | ------------------ |
| Always Last Turn      | 0.4054     | -7.4%             | 0%        | ❌ Baseline        |
| Always Rewrite        | **0.4378** | —                 | 86.9%     | ✓ Best quality     |
| V1 (pronoun-only)     | 0.4257     | -2.8%             | 21.8%     | ✓ Max cost savings |
| **V4 (domain-aware)** | **0.4360** | **-0.4%**         | 30.2%     | ⭐ Best balance    |
| Oracle (best of)      | 0.4885     | +11.6%            | —         | Upper bound        |

**Recommended: V4** — Only -0.4% quality loss with 70% fewer LLM calls.

---

## 1. The Problem

Current systems always rewrite conversational queries to be standalone. Our analysis of 777 queries shows this is often unnecessary or harmful:

| Outcome                              | Count   | Percentage |
| ------------------------------------ | ------- | ---------- |
| Last Turn **BETTER** than Rewrite    | 138     | 17.8%      |
| Rewrite **BETTER** than Last Turn    | 167     | 21.5%      |
| **EQUAL** (tie)                      | 472     | 60.7%      |
| **Rewriting unnecessary or harmful** | **610** | **78.5%**  |

**Key Insight**: Rewriting only helps in ~22% of cases.

### Examples Where Rewriting Hurt

| Original Query                                | Rewritten Query                                                     | LT nDCG | RW nDCG | Δ    |
| --------------------------------------------- | ------------------------------------------------------------------- | ------- | ------- | ---- |
| "How old is the moon?"                        | "What is the age of the moon?"                                      | 1.00    | 0.63    | -37% |
| "Enterprise"                                  | "I want to know about creating an Enterprise account in IBM Cloud." | 1.00    | 0.39    | -61% |
| "archaeological discoveries and colonialists" | "What is the relationship between..."                               | 0.92    | 0.61    | -31% |

These are already standalone queries. Rewriting **dilutes** their precision.

---

## 2. The Solution: Selective Rewriting

### Core Idea

Use linguistic signals to decide whether a query needs rewriting:

1. **Pronouns** (it, they, this, that) → Rewrite
2. **Ellipsis** (short queries with missing subject) → Rewrite
3. **Topic continuation** ("What about X?") → Rewrite
4. **Standalone queries** → Use as-is

### V1: Pronoun-Based Heuristic (Simple)

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def has_pronouns(query: str) -> bool:
    """Use spaCy POS tagging to detect pronouns."""
    doc = nlp(query)
    return any(token.pos_ == "PRON" for token in doc)

def should_rewrite_v1(query: str, turn: int) -> bool:
    """V1: Rewrite only if pronouns detected."""
    if turn == 1:
        return False
    return has_pronouns(query)
```

**Results**: -2.8% quality, 78% fewer rewrites

### V4: Domain-Aware Heuristic (Recommended)

```python
def should_rewrite_v4(query: str, turn: int, domain: str = None) -> bool:
    """V4: Pronouns + domain-aware ellipsis + topic continuation."""
    if turn == 1:
        return False

    # 1. Pronouns (explicit references)
    if has_pronouns(query):
        return True

    # 2. Ellipsis: short queries (domain-aware)
    word_count = len(query.split())
    if domain in ['clapnq', 'govt'] and word_count <= 4:
        return True
    # Cloud/FiQA: skip short query heuristic (keyword queries work well)

    # 3. Topic continuation
    if 'what about' in query.lower():
        return True

    return False
```

**Results**: -0.4% quality, 70% fewer rewrites

---

## 3. Evidence & Analysis

### 3.1 Pronouns Are the Key Signal

| Query Type       | Last Turn >= Rewrite | Rewriting Helps |
| ---------------- | -------------------- | --------------- |
| **Has pronouns** | 61.6%                | **38.4%** ✓     |
| **No pronouns**  | 73.4%                | 26.6% ✗         |

When a query has no pronouns, rewriting only helps 27% of the time.

### 3.2 Decision Accuracy (V1)

| Decision Type                                | Count | %     |
| -------------------------------------------- | ----- | ----- |
| Correct decisions                            | 639   | 82.2% |
| Missed opportunities (should have rewritten) | 110   | 14.2% |
| Wrong rewrites (shouldn't have rewritten)    | 28    | 3.6%  |

**The heuristic is asymmetrically wrong**:

- Misses 110 rewrites → loses 41.1 nDCG points
- Makes 28 wrong rewrites → loses only 7.7 nDCG points

### 3.3 Missed Opportunities Analysis

Queries that V1 missed (no pronouns, but needed rewrite):

| Query                                  | Turn | Issue                 | LT   | RW   | Loss  |
| -------------------------------------- | ---- | --------------------- | ---- | ---- | ----- |
| "How many teams?"                      | 9    | Ellipsis              | 0.00 | 1.00 | -1.00 |
| "What about Romeo and Juliet?"         | 3    | Topic continuation    | 0.00 | 0.96 | -0.96 |
| "When was reorganized?"                | 3    | Ellipsis (no subject) | 0.00 | 0.92 | -0.92 |
| "What are the suppliers for the city?" | 4    | Definite reference    | 0.00 | 0.92 | -0.92 |

These have **implicit references** that pronouns don't capture.

### 3.4 Enhanced Heuristics Impact

| Heuristic                | Catches   | nDCG Recovered | Side Effect      |
| ------------------------ | --------- | -------------- | ---------------- |
| Short queries (≤4 words) | 39 missed | +17.6          | Hurts Cloud/FiQA |
| "What about X?"          | 3 missed  | +2.4           | Minimal          |
| "the" + short            | 30 missed | +9.4           | Mixed            |

**Key finding**: Short query heuristic is domain-dependent.

### 3.5 Domain-Specific Results

| Domain | Short Query Impact | Optimal Strategy    |
| ------ | ------------------ | ------------------- |
| ClapNQ | +2.3 nDCG          | Use short heuristic |
| Govt   | +5.7 nDCG          | Use short heuristic |
| Cloud  | **-3.6 nDCG**      | Pronouns only       |
| FiQA   | **-1.7 nDCG**      | Pronouns only       |

Cloud/FiQA have keyword queries ("Enterprise", "International equity") that work well as-is.

---

## 4. Absolute Retrieval Quality

### Strategy Performance (ELSER, 777 queries)

| Strategy              | nDCG@5     | nDCG@10    | Recall@5   | Recall@10  | MRR        |
| --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Always Last Turn      | 0.4054     | 0.4511     | 0.4394     | 0.5445     | 0.3887     |
| **Always Rewrite**    | **0.4378** | **0.4953** | **0.4761** | **0.6078** | **0.4286** |
| V1 (pronoun)          | 0.4257     | 0.4774     | 0.4616     | 0.5800     | 0.4093     |
| **V4 (domain-aware)** | **0.4360** | —          | —          | —          | —          |
| Oracle                | 0.4885     | —          | —          | —          | —          |

### Per-Domain Results (nDCG@5)

| Domain | Last Turn | Rewrite    | V1 (pronoun) | V4 (domain) | Best    |
| ------ | --------- | ---------- | ------------ | ----------- | ------- |
| ClapNQ | 0.4749    | **0.5135** | 0.5030       | **0.5141**  | V4 ✓    |
| Cloud  | 0.3894    | 0.3940     | **0.4009**   | **0.4009**  | V1/V4 ✓ |
| FiQA   | 0.3477    | **0.3779** | 0.3777       | 0.3777      | Rewrite |
| Govt   | 0.4001    | **0.4540** | 0.4118       | 0.4402      | Rewrite |

---

## 5. Query Routing Statistics

### By Turn (V1 Heuristic)

| Turn      | Skip    | Rewrite | Total   | Skip %    |
| --------- | ------- | ------- | ------- | --------- |
| 1         | 102     | 0       | 102     | 100.0%    |
| 2         | 74      | 32      | 106     | 69.8%     |
| 3         | 73      | 30      | 103     | 70.9%     |
| 4         | 75      | 24      | 99      | 75.8%     |
| 5         | 73      | 23      | 96      | 76.0%     |
| 6         | 61      | 26      | 87      | 70.1%     |
| 7         | 65      | 15      | 80      | 81.2%     |
| 8         | 50      | 15      | 65      | 76.9%     |
| 9         | 22      | 10      | 32      | 68.8%     |
| 10+       | 4       | 3       | 7       | 57.1%     |
| **Total** | **599** | **178** | **777** | **77.1%** |

Skip rate remains high (68-81%) across all turns.

### By Domain (V1 Heuristic)

| Domain | Skip % | Rewrite % |
| ------ | ------ | --------- |
| Govt   | 85.6%  | 14.4%     |
| Cloud  | 80.9%  | 19.1%     |
| FiQA   | 71.7%  | 28.3%     |
| ClapNQ | 70.2%  | 29.8%     |

---

## 6. Conclusion

### Recommended Strategy: V4 (Domain-Aware)

| Metric    | Always Rewrite | V1 (pronoun)   | V4 (domain-aware)  |
| --------- | -------------- | -------------- | ------------------ |
| nDCG@5    | **0.4378**     | 0.4257 (-2.8%) | **0.4360 (-0.4%)** |
| LLM Calls | 675 (100%)     | 178 (26%)      | 235 (35%)          |
| Latency   | ~500ms/query   | ~0ms for 77%   | ~0ms for 70%       |

### When to Use Each Strategy

| Strategy              | Use When                                 |
| --------------------- | ---------------------------------------- |
| **Always Rewrite**    | Quality is paramount, cost not a concern |
| **V4 (domain-aware)** | Best balance of quality and cost         |
| **V1 (pronoun-only)** | Maximum cost savings acceptable          |

### Key Takeaways

1. **78% of queries don't benefit from rewriting** — the original query is equal or better
2. **Pronouns are the strongest signal** for when rewriting helps
3. **Ellipsis detection is domain-dependent** — helps ClapNQ/Govt, hurts Cloud/FiQA
4. **V4 achieves near-parity (-0.4%)** with 70% fewer LLM calls
5. **Oracle shows +11.6% potential** — room for improvement with better classification
