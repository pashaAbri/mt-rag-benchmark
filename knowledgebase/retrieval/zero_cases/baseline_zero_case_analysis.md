# Zero-Case Analysis: Baseline Retrieval Strategy Failures

## Overview

This document analyzes **cases where all three baseline retrieval strategies fail to retrieve any relevant documents** (R@1 = R@3 = R@5 = R@10 = 0). These represent the "hardest" cases in the MT-RAG benchmark where standard approaches completely fail.

**Source**: Analysis from `scripts/ideas/retrieval_tasks/oracle-v2/new-strategies/`

---

## Key Findings Summary

| Metric                          | Value                                                        |
| ------------------------------- | ------------------------------------------------------------ |
| **Total Zero-Score Cases**      | 98 out of 777 tasks (12.6%)                                  |
| **Retrievers Analyzed**         | ELSER, BM25, BGE                                             |
| **Strategies Analyzed**         | Last Turn, Query Rewrite, Questions                          |
| **Complete Failure Rate**       | All 9 retriever-strategy combinations return 0 relevant docs |
| **MonoT5 Helps Zero Cases?**    | ❌ **No** — 0% recovery rate                                  |
| **MonoT5 Hurts Some Cases?**    | ⚠️ **Yes** — 38 cases went from positive → zero              |

---

## 1. Domain Distribution

Zero cases are distributed relatively evenly across domains, with Cloud having slightly more failures:

| Domain     | Zero Cases | Percentage |
| ---------- | ---------- | ---------- |
| **Cloud**  | 28         | 28.6%      |
| **FiQA**   | 25         | 25.5%      |
| **Govt**   | 23         | 23.5%      |
| **ClapNQ** | 22         | 22.4%      |

**Observation**: Cloud documentation appears slightly more challenging for retrieval, possibly due to technical jargon and specialized terminology that may not match well with conversational queries.

---

## 2. Turn Distribution

The vast majority of zero-score cases occur in **later turns**, not first turns:

| Turn Position   | Count | Percentage |
| --------------- | ----- | ---------- |
| **First Turn**  | 5     | 5.1%       |
| **Later Turns** | 93    | 94.9%      |

### Turn-by-Turn Breakdown

| Turn   | Count | Percentage |
| ------ | ----- | ---------- |
| Turn 1 | 5     | 5.1%       |
| Turn 2 | 5     | 5.1%       |
| Turn 3 | 12    | 12.2%      |
| Turn 4 | 8     | 8.2%       |
| Turn 5 | 13    | 13.3%      |
| Turn 6 | 20    | 20.4%      |
| Turn 7 | 13    | 13.3%      |
| Turn 8 | 17    | 17.3%      |
| Turn 9 | 5     | 5.1%       |

**Key Insight**: Zero cases peak at **turns 6-8**, suggesting that:

- Context dependency increases as conversations progress
- Query rewrites may lose critical information from earlier turns
- Pronoun/reference resolution becomes more challenging deep in conversations

---

## 3. Answerability Distribution

Despite being zero-score cases, **88.8% are marked as ANSWERABLE**:

| Answerability  | Count | Percentage |
| -------------- | ----- | ---------- |
| **ANSWERABLE** | 87    | 88.8%      |
| **PARTIAL**    | 11    | 11.2%      |

**Critical Insight**: These are not unanswerable questions—the relevant documents exist in the corpus, but all retrieval strategies fail to find them. This represents a **retrieval gap**, not a data gap.

---

## 4. Question Type Distribution

| Question Type       | Count | Percentage |
| ------------------- | ----- | ---------- |
| **Factoid**         | 25    | 25.5%      |
| **Explanation**     | 16    | 16.3%      |
| **Keyword**         | 14    | 14.3%      |
| **Summarization**   | 10    | 10.2%      |
| **Non-Question**    | 9     | 9.2%       |
| **How-To**          | 9     | 9.2%       |
| **Opinion**         | 7     | 7.1%       |
| **Comparative**     | 4     | 4.1%       |
| **Troubleshooting** | 3     | 3.1%       |
| **Composite**       | 1     | 1.0%       |

**Observations**:

- **Factoid queries** (25.5%) are surprisingly challenging—likely due to implicit context requirements
- **Non-Questions** (9.2%) represent statements/comments where retrieval intent is unclear
- **Keyword queries** (14.3%) may lack sufficient context for disambiguation

---

## 5. Multi-Turn Type Distribution

| Multi-Turn Type      | Count | Percentage |
| -------------------- | ----- | ---------- |
| **Follow-up**        | 73    | 74.5%      |
| **Clarification**    | 20    | 20.4%      |
| **N/A** (first turn) | 5     | 5.1%       |

**Key Insight**: **Follow-up queries** dominate zero cases (74.5%), suggesting:

- These queries heavily depend on prior conversation context
- Query rewrite may not adequately capture the topic shift or continuation
- Entity/concept references from earlier turns are not being resolved

---

## 6. Query Characteristics

| Characteristic           | Value      |
| ------------------------ | ---------- |
| **Average Query Length** | 7.8 words  |
| **With Question Mark**   | 60 (61.2%) |
| **With WH-word**         | 38 (38.8%) |

**Observations**:

- Queries are relatively **short** (7.8 words on average)
- ~39% lack explicit WH-words, making them more implicit/conversational
- Short queries with implicit references are the hardest to retrieve

---

## 7. Sample Zero-Score Cases

### ClapNQ Examples

1. **"That is too bad. Although the movement was non-violent, some ended up in violence, right?"**

   - Turn 6, Follow-up, Summarization
   - **Why it fails**: Requires resolving "the movement" from conversation history

2. **"By how much did the cotton gin increase production?"**

   - Turn 3, Follow-up, Summarization
   - **Why it fails**: "cotton gin" topic must be inferred from context

3. **"I like ice hockey as my sons were great players..."**
   - Turn 9, Non-Question, Follow-up
   - **Why it fails**: Not a question—unclear retrieval intent

### Cloud Examples

1. **"does IBM offer document databases?"**

   - Turn 1 (First Turn!), Factoid
   - **Why it fails**: Possibly vocabulary mismatch with corpus (e.g., "Cloudant" vs "document database")

2. **"i am having hard time finding web chat whenever I was trying to use"**

   - Turn 6, Troubleshooting, Follow-up
   - **Why it fails**: Incomplete query, requires context about which product

3. **"Add user identity information"**
   - Turn 7, Keyword, Follow-up
   - **Why it fails**: Very short keyword query, no context about the product/feature

### FiQA Examples

1. **"What is the ideal formula to evaluate a company?"**

   - Turn 7, Summarization, Follow-up
   - **Why it fails**: "a company" may refer to a specific company discussed earlier

2. **"if it is free, then I should make a second account for my business. how can I create?"**

   - Turn 3, How-To, Follow-up
   - **Why it fails**: "it" reference unresolved, incomplete sentence

3. **"can you let me know advantages of using two bank accounts?"**
   - Turn 7, Explanation, Follow-up
   - **Why it fails**: May require specific context about user's situation

### Govt Examples

1. **"Does Clipper have an antennae?"**

   - Turn 6, Factoid, Follow-up
   - **Why it fails**: "Clipper" is ambiguous without prior context (spacecraft?)

2. **"Does Clipper fly by Europe?"**

   - Turn 8, Factoid, Follow-up
   - **Why it fails**: Same ambiguity—"Europe" might refer to Europa moon

3. **"Who is responsible for covering my expenses?"**
   - Turn 4, Explanation, Follow-up
   - **Why it fails**: "my expenses" requires context about the situation

---

## 8. Failure Pattern Analysis

### Primary Failure Modes

1. **Context-Dependent References** (~60% of cases)

   - Pronouns: "it", "this", "that", "the movement"
   - Implicit entities: "the company", "my account", "Clipper"
   - Topic continuation without explicit mention

2. **Short/Ambiguous Queries** (~25% of cases)

   - Under 5 words with insufficient context
   - Keyword-style queries lacking specificity
   - Non-questions (statements/comments)

3. **Vocabulary Mismatch** (~15% of cases)
   - User terminology vs. corpus terminology
   - Informal language vs. technical documentation
   - Missing domain-specific synonyms

### Why All Three Strategies Fail

| Strategy          | Failure Mode                                              |
| ----------------- | --------------------------------------------------------- |
| **Last Turn**     | No context → references unresolved                        |
| **Query Rewrite** | Context may be wrong/incomplete → hallucinated resolution |
| **Questions**     | Aggregates all questions → dilutes specific intent        |

---

## 9. MonoT5 Reranking Impact Analysis

### Does MonoT5 Fusion+Reranking Help Zero Cases?

**Short answer: No.** MonoT5 3-strategy fusion with reranking **cannot recover any of the 98 zero-score cases**.

| Metric                        | Value           |
| ----------------------------- | --------------- |
| Zero cases helped by MonoT5   | **0 (0.0%)**    |
| Zero cases still at zero      | **98 (100.0%)** |

#### Why Reranking Cannot Help

```
Zero-Score Case Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Last Turn retrieves 0 relevant docs                         │
│ 2. Query Rewrite retrieves 0 relevant docs                     │
│ 3. Questions retrieves 0 relevant docs                         │
│    ↓                                                            │
│ 4. Fusion combines all 3 pools → Still 0 relevant docs         │
│    ↓                                                            │
│ 5. MonoT5 reranks the pool → Cannot surface non-existent docs  │
└─────────────────────────────────────────────────────────────────┘
```

**Conclusion**: Zero-score cases represent a **first-stage retrieval problem**, not a reranking problem. These cases need better query formulation or alternative retrieval strategies—reranking an empty/irrelevant pool cannot help.

---

### Does MonoT5 Hurt Any Cases?

**Yes.** While MonoT5 provides a net positive effect overall, it **hurts 25.9% of cases** and even **creates 38 new zero-score cases** from previously successful retrievals.

| Impact Category              | Count | Percentage |
| ---------------------------- | ----- | ---------- |
| **HELPED** (nDCG@5 > +0.01)  | 257   | 33.1%      |
| **HURT** (nDCG@5 < -0.01)    | 201   | 25.9%      |
| **SAME** (within ±0.01)      | 319   | 41.1%      |

#### Overall Net Effect

| Metric                          | Value                  |
| ------------------------------- | ---------------------- |
| Average nDCG@5 change           | **+0.027** (positive)  |
| Average improvement (helped)    | +0.284 nDCG@5          |
| Average degradation (hurt)      | -0.258 nDCG@5          |
| **Cases that BECAME zero**      | **38** (from positive) |

#### Worst Hurt Cases (Perfect → Zero)

| Task ID                               | Baseline nDCG@5 | Reranked nDCG@5 | Δ      |
| ------------------------------------- | --------------- | --------------- | ------ |
| `694e275f1a01ad0e8ac448ad809f7930<::>7` | 1.000           | 0.000           | -1.000 |
| `fd99b316e5e64f19ff938598aea9b285<::>9` | 1.000           | 0.000           | -1.000 |
| `1be66272113492407e814eaf21a761d4<::>4` | 1.000           | 0.000           | -1.000 |
| `c6c3b02ca32795af64c903dd76700517<::>4` | 1.000           | 0.000           | -1.000 |
| `5f9ccf0a4ff691fc482432af64cc3c9d<::>7` | 0.920           | 0.000           | -0.920 |

#### Why MonoT5 Sometimes Hurts

1. **Query-Document Mismatch**: MonoT5 uses the `rewrite` query for scoring, but the relevant doc may have been retrieved by `lastturn` or `questions` with different phrasing

2. **Noise Introduction**: Adding documents from weaker strategies can push relevant documents below the top-k cutoff

3. **Cross-Encoder Misranking**: MonoT5 may not understand domain-specific relevance (especially for technical Cloud/Govt docs)

---

### Key Takeaways

1. **Zero cases are unreachable by reranking** — they require fundamentally better retrieval queries

2. **MonoT5 is a trade-off**: +33% helped, -26% hurt, net positive but volatile

3. **38 cases were destroyed** by fusion+reranking (went from successful to zero)

4. **Strategy selection** may be more valuable than fusion — knowing *when* to use each strategy could avoid the hurt cases

---

### Priority Cases to Address

Focus on:

- **Later turns** (Turns 5-8) where context dependency is highest
- **Follow-up queries** which represent 74.5% of zero cases
- **Factoid and Explanation** question types

---
