## The Three Metrics: What They Measure and How

### 1. **RL_F (Reference-Less Faithfulness)** — _"Is it grounded?"_

| Aspect               | Description                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| **What it measures** | Whether the LLM's response is faithful to the **retrieved passages**                            |
| **Calculation**      | LLM judge (RAGAS framework) analyzes the response against the passages to detect hallucinations |
| **Inputs**           | Response + Retrieved passages (does **NOT** need the reference answer)                          |
| **Range**            | 0.0 – 1.0 (higher = more faithful)                                                              |

**Intuition**: If you gave the model these documents, did it make up stuff that wasn't in them? A high RL_F means the model "stayed in its lane" and only said things supported by the context.

**Key insight**: This metric is **reference-less** — it doesn't care if the answer is _correct_, only that it's _grounded_ in the provided passages.

---

### 2. **RB_llm (Reference-Based LLM Judge)** — _"Is it a good answer?"_

| Aspect               | Description                                                                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What it measures** | Overall response quality compared to a human-written reference answer                                                                                                 |
| **Calculation**      | LLM judge evaluates three properties: **Faithfulness** (to docs + conversation), **Appropriateness** (relevant to the question), **Completeness** (includes key info) |
| **Inputs**           | Response + Reference answer + Passages + Conversation                                                                                                                 |
| **Range**            | 0.0 – 1.0 (normalized from 1-10 scale)                                                                                                                                |

**Intuition**: If a human expert wrote the ideal answer, how close is the model's response? This is the most holistic metric — it captures whether the answer is actually useful.

---

### 3. **RB_agg (Reference-Based Algorithmic Aggregate)** — _"How similar to the reference?"_

| Aspect               | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| **What it measures** | Textual/semantic similarity to the reference answer                          |
| **Calculation**      | Harmonic mean of three components:                                           |
|                      | • **BERTScore-Recall**: Semantic overlap with reference (completeness proxy) |
|                      | • **ROUGE-L**: Phrase-level overlap with reference (appropriateness proxy)   |
|                      | • **BERTKPrec**: BERT similarity to passages (faithfulness proxy)            |
| **Inputs**           | Response + Reference answer + Passages                                       |
| **Range**            | 0.0 – 1.0                                                                    |

**Formula**:

```python
recall = (BertscoreR + 1) / 2
rouge = RougeL_stemFalse
extractiveness = (BertKPrec + 1) / 2

RB_agg = harmonic_mean(recall, rouge, extractiveness)
```

**Intuition**: A cheaper, faster, algorithm-based approximation of answer quality. No LLM calls needed.

---

### H-Mean: Why Combine Them?

Your H-Mean = harmonic mean of (RL_F, RB_llm, RB_agg) ensures a system must perform well on **all three dimensions**:

| Metric | What it penalizes                            |
| ------ | -------------------------------------------- |
| RL_F   | Hallucination (making stuff up)              |
| RB_llm | Poor answer quality (incomplete, irrelevant) |
| RB_agg | Not matching the reference text              |

Harmonic mean is harsh — if any single metric is low, the H-Mean is pulled down significantly.
