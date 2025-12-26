# Targeted Rewrite: Controlled Experiment with Mixtral

## Purpose

This experiment isolates the effect of **context filtering** by using the same LLM (Mixtral 8x7B) as the paper baseline. The original targeted rewrite experiment used Claude Sonnet 4.5, which introduced a confound: we couldn't determine if performance gains came from better context filtering or simply from using a better LLM.

## Experimental Design

| Experiment | LLM | Context Strategy |
|------------|-----|------------------|
| **Baseline** (paper) | Mixtral 8x7B | Full conversation history |
| **Targeted Rewrite** | Mixtral 8x7B | Filtered history (similarity ≥ 0.3) |

### Targeted Rewrite Parameters

- **Embedding Model**: `all-MiniLM-L6-v2`
- **LLM Model**: `Mixtral 8x7B Instruct` (via Together AI)
- **Similarity Threshold**: 0.3
- **Max Relevant Turns**: 5
- **Include Last Turn**: Always (regardless of similarity)

---

## Results: ELSER Retrieval

### ClapNQ

| Metric | Baseline | Targeted | Δ |
|--------|----------|----------|---|
| R@1 | 0.209 | 0.211 | +0.2% |
| R@3 | 0.424 | 0.445 | +2.1% |
| R@5 | 0.552 | 0.569 | +1.7% |
| R@10 | 0.701 | 0.712 | +1.1% |
| nDCG@1 | 0.524 | 0.514 | -1.0% |
| nDCG@3 | 0.470 | 0.482 | +1.2% |
| nDCG@5 | 0.513 | 0.527 | +1.4% |
| nDCG@10 | 0.578 | 0.591 | **+1.3%** ✅ |

### Cloud (IBM Cloud)

| Metric | Baseline | Targeted | Δ |
|--------|----------|----------|---|
| R@1 | 0.179 | 0.131 | -4.8% |
| R@3 | 0.353 | 0.287 | -6.6% |
| R@5 | 0.430 | 0.358 | -7.2% |
| R@10 | 0.528 | 0.438 | -9.0% |
| nDCG@1 | 0.378 | 0.293 | -8.5% |
| nDCG@3 | 0.365 | 0.296 | -6.9% |
| nDCG@5 | 0.394 | 0.323 | -7.1% |
| nDCG@10 | 0.438 | 0.356 | **-8.2%** ❌ |

### FiQA

| Metric | Baseline | Targeted | Δ |
|--------|----------|----------|---|
| R@1 | 0.163 | 0.140 | -2.3% |
| R@3 | 0.310 | 0.289 | -2.1% |
| R@5 | 0.402 | 0.375 | -2.7% |
| R@10 | 0.536 | 0.491 | -4.5% |
| nDCG@1 | 0.389 | 0.339 | -5.0% |
| nDCG@3 | 0.344 | 0.315 | -2.9% |
| nDCG@5 | 0.378 | 0.347 | -3.1% |
| nDCG@10 | 0.436 | 0.396 | **-4.0%** ❌ |

### Govt

| Metric | Baseline | Targeted | Δ |
|--------|----------|----------|---|
| R@1 | 0.194 | 0.179 | -1.5% |
| R@3 | 0.392 | 0.404 | +1.2% |
| R@5 | 0.508 | 0.499 | -0.9% |
| R@10 | 0.651 | 0.620 | -3.1% |
| nDCG@1 | 0.413 | 0.418 | +0.5% |
| nDCG@3 | 0.407 | 0.416 | +0.9% |
| nDCG@5 | 0.454 | 0.448 | -0.6% |
| nDCG@10 | 0.517 | 0.502 | **-1.5%** ❌ |

---

## Aggregate Summary

| Metric | Baseline | Targeted | Δ |
|--------|----------|----------|---|
| R@1 | 0.186 | 0.165 | -2.1% |
| R@3 | 0.370 | 0.356 | -1.4% |
| R@5 | 0.473 | 0.450 | -2.3% |
| R@10 | 0.604 | 0.565 | -3.9% |
| nDCG@1 | 0.426 | 0.391 | -3.5% |
| nDCG@3 | 0.397 | 0.377 | -2.0% |
| nDCG@5 | 0.435 | 0.411 | -2.4% |
| nDCG@10 | 0.492 | 0.461 | **-3.1%** |

---

## Key Findings

### 1. Context Filtering Hurts Performance

When using the **same LLM** (Mixtral 8x7B), the targeted rewrite strategy **degrades** retrieval performance:

| Domain | nDCG@10 Change |
|--------|----------------|
| ClapNQ | +1.3% ✅ |
| Cloud | -8.2% ❌ |
| FiQA | -4.0% ❌ |
| Govt | -1.5% ❌ |
| **Average** | **-3.1%** |

### 2. Only ClapNQ Benefits

ClapNQ is the only domain where filtering helps. This may be because:
- ClapNQ has more open-domain, general knowledge questions
- Topic shifts are more common, making irrelevant context more likely
- Questions are less dependent on technical terminology from earlier turns

### 3. Cloud Suffers Most

Cloud (IBM Cloud documentation) shows the largest degradation (-8.2%). Reasons:
- Technical terminology requires full context for disambiguation
- API names, service names, and configuration details often span multiple turns
- Filtering removes crucial domain-specific context

### 4. The Hypothesis Is Not Supported

**Original Hypothesis**: Filtering irrelevant conversation history will produce more focused rewrites and improve retrieval.

**Finding**: The hypothesis is **not supported**. With the same LLM:
- 3 out of 4 domains show degradation
- Average performance drops by 3.1%
- The simple similarity-based filtering removes useful context

---

## Why Does Filtering Hurt?

### 1. Loss of Disambiguating Context

Even turns with low semantic similarity may contain important context:
- Entity introductions ("I'm asking about the VPC cluster")
- Constraint specifications ("For the free tier only")
- Domain anchoring ("In IBM Cloud...")

### 2. Mixtral Needs More Context

Smaller models like Mixtral 8x7B may benefit from more context to understand the conversation, while larger models like Claude can infer from less.

### 3. Similarity Threshold Too Aggressive

The 0.3 threshold may filter out turns that are topically related but lexically different. A turn discussing "pricing plans" may have low similarity to a question about "costs" even though they're related.

---

## Recommendations

1. **Do not use simple similarity-based context filtering** for query rewriting
2. **Include full conversation history** when using smaller LLMs like Mixtral
3. **Consider LLM quality** as the primary lever for improving query rewriting
4. **If filtering is needed**, explore more sophisticated approaches:
   - Coreference-aware filtering (keep turns that introduce referenced entities)
   - Domain-specific filtering (keep turns with technical terms)
   - Hybrid approaches that combine similarity with other signals

---

## Source

Experiment code: `scripts/ideas/retrieval_tasks/targeted_rewrite_with_mixtral/`

Results:
- Rewritten queries: `intermediate/targeted_rewrite_mixtral_{domain}.jsonl`
- Retrieval results: `retrieval_results/targeted_rewrite_mixtral_{domain}_elser.jsonl`
- Evaluations: `retrieval_results/targeted_rewrite_mixtral_{domain}_elser_evaluated_aggregate.csv`

