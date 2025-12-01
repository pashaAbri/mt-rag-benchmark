## What QPP-GenRE Does

**Core Problem**: Query Performance Prediction (QPP) aims to estimate how well a search system will perform for a query _without_ human relevance judgments. Traditional QPP methods output a single scalar score that doesn't map to specific IR metrics and isn't interpretable.

**Their Solution**: Decompose QPP into predicting relevance for each document individually, then compute any IR metric from those predictions.

### Method Overview

1. **Relevance Judgment Generation**: Use an LLM to predict "Relevant" or "Irrelevant" for each query-document pair in a ranked list
2. **Metric Calculation**: Compute RR@10, nDCG@10, Precision@k, etc. from the predicted labels
3. **Fine-tuning**: Train open-source LLMs (Llama, Mistral) on human relevance labels using QLoRA

### Key Findings

| Finding                                  | Implication                                          |
| ---------------------------------------- | ---------------------------------------------------- |
| Fine-tuned 3B model > 70B few-shot model | Small fine-tuned models are cost-effective           |
| State-of-the-art on TREC-DL 19–22        | Works for both BM25 and neural rankers (ANCE, TAS-B) |
| Judging depth 10–200 is sufficient       | Don't need to judge entire corpus                    |
| Generalizes to conversational search     | Domain-transferable                                  |
| Interpretable errors                     | Can trace QPP failures to specific FP/FN judgments   |

---

## How This Applies to My Research

### 1. Post-Retrieval Quality Prediction for Fallback Decisions

Our current framework uses ELSER confidence scores, but we identified the **Keyword paradox**: high confidence but poor retrieval. QPP-GenRE could replace or augment this:

```
Current:  Keyword + High Confidence → Distrust, force Rewrite
Proposed: Keyword + QPP-GenRE predicts low nDCG → Trigger fallback
```

QPP-GenRE predicts _actual retrieval quality_, not just confidence. This would give calibrated post-hoc validation.

---

### 2. Strategy Selection via Predicted Retrieval Quality

Instead of routing based solely on query type classification, run all three strategies through first-stage retrieval, then use QPP-GenRE to predict which will perform best:

| Strategy  | QPP-GenRE Predicted nDCG@10 | Decision |
| --------- | --------------------------- | -------- |
| Last Turn | 0.42                        |          |
| Rewrite   | 0.58                        | ← Select |
| Questions | 0.51                        |          |

This converts the oracle routing problem into a **per-query optimization** rather than a static rule-based system.

---

### 3. Domain-Specific Quality Estimation

Our analysis found GOVT has consistently low confidence scores (15–20 range), indicating potential domain mismatch. QPP-GenRE could be fine-tuned per domain:

| Domain | Fine-tuning Data    | Use Case                                         |
| ------ | ------------------- | ------------------------------------------------ |
| FIQA   | Financial Q&A pairs | Better calibration for Rewrite-preferring domain |
| GOVT   | Policy documents    | Accurate QPP for low-confidence queries          |
| CLOUD  | Technical docs      | Validate Questions strategy dominance            |

This addresses the domain-specific confidence calibration issues we identified.

---

### 4. Interpretable Routing Errors

Our current framework can say "Routing to Rewrite failed" but not _why_. QPP-GenRE produces document-level relevance predictions, enabling:

- **Error Attribution**: "Top-3 documents predicted irrelevant → strategy failed to surface relevant content"
- **Targeted Improvement**: If false negatives dominate, improve recall; if false positives dominate, improve precision
- **Query-Specific Diagnosis**: Link routing failures to specific retrieval patterns

---

### 5. Integration with Confidence Calibration

The paper shows that QPP-GenRE's relevance predictions have interpretable error patterns (Table 8):

| Class      | Precision | Recall | Pattern                                    |
| ---------- | --------- | ------ | ------------------------------------------ |
| Relevant   | 0.58      | 0.30   | Under-predicts relevance (false negatives) |
| Irrelevant | 0.78      | 0.92   | Accurate                                   |

This aligns with our Keyword paradox: ELSER is overconfident, and QPP-GenRE could provide a second opinion that's biased toward _conservative_ relevance prediction—exactly what's needed for high-confidence, low-retrieval scenarios.

---

### Proposed Integration into Oracle Router

```
1. Query arrives
   ↓
2. Query-type classifier → Primary strategy selection
   ↓
3. First-stage retrieval (ELSER)
   ↓
4. QPP-GenRE predicts nDCG@10 for top-n results
   ↓
5. If predicted nDCG < threshold:
   → Trigger fallback (alternative strategy or query expansion)
   ↓
6. Return results with QPP-based confidence score
```

This adds a **learned quality gate** after retrieval, informed by document-level relevance predictions rather than raw ELSER scores.

---

### Summary: What QPP-GenRE Adds

| Our Current Approach               | With QPP-GenRE                                |
| ---------------------------------- | --------------------------------------------- |
| Confidence = ELSER score           | Confidence = predicted nDCG@10                |
| Static routing rules               | Dynamic per-query optimization                |
| Opaque failures                    | Interpretable relevance errors                |
| Single-domain calibration          | Domain-specific fine-tuning                   |
| Fallback based on score thresholds | Fallback based on predicted retrieval quality |

The key insight is that QPP-GenRE transforms **confidence scores** (which can be miscalibrated) into **predicted retrieval quality** (which is directly actionable for routing decisions).
