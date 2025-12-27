# Comprehensive Comparison: Context Filtering vs Full History

This document provides a complete comparison of all four experimental conditions, isolating the effects of **LLM quality** and **context filtering**.

## Experimental Conditions

| Condition            | LLM           | Context Strategy | Description                   |
| -------------------- | ------------- | ---------------- | ----------------------------- |
| **Mixtral Full**     | Mixtral 8x7B  | Full history     | Baseline with live pipeline   |
| **Mixtral Filtered** | Mixtral 8x7B  | Similarity ≥ 0.3 | Targeted rewrite with Mixtral |
| **Sonnet Full**      | Claude Sonnet | Full history     | Baseline for Sonnet           |
| **Sonnet Filtered**  | Claude Sonnet | Similarity ≥ 0.3 | Targeted rewrite with Sonnet  |

---

## Results

### Recall Metrics

| Domain | Condition            | R@1   | R@3   | R@5   | R@10  |
| ------ | -------------------- | ----- | ----- | ----- | ----- |
| All    | **Mixtral Full**     | 0.163 | 0.352 | 0.449 | 0.562 |
|        | **Mixtral Filtered** | 0.164 | 0.352 | 0.449 | 0.566 |
|        | **Sonnet Full**      | 0.191 | 0.380 | 0.488 | 0.607 |
|        | **Sonnet Filtered**  | 0.187 | 0.380 | 0.483 | 0.602 |
| ClapNQ | **Mixtral Full**     | 0.198 | 0.445 | 0.558 | 0.693 |
|        | **Mixtral Filtered** | 0.183 | 0.434 | 0.532 | 0.685 |
|        | **Sonnet Full**      | 0.220 | 0.445 | 0.580 | 0.726 |
|        | **Sonnet Filtered**  | 0.222 | 0.454 | 0.588 | 0.732 |
| Cloud  | **Mixtral Full**     | 0.127 | 0.267 | 0.332 | 0.435 |
|        | **Mixtral Filtered** | 0.137 | 0.283 | 0.373 | 0.460 |
|        | **Sonnet Full**      | 0.160 | 0.321 | 0.414 | 0.505 |
|        | **Sonnet Filtered**  | 0.155 | 0.312 | 0.384 | 0.479 |
| FiQA   | **Mixtral Full**     | 0.149 | 0.278 | 0.362 | 0.480 |
|        | **Mixtral Filtered** | 0.128 | 0.280 | 0.347 | 0.478 |
|        | **Sonnet Full**      | 0.175 | 0.324 | 0.414 | 0.532 |
|        | **Sonnet Filtered**  | 0.169 | 0.317 | 0.400 | 0.537 |
| Govt   | **Mixtral Full**     | 0.175 | 0.400 | 0.523 | 0.618 |
|        | **Mixtral Filtered** | 0.202 | 0.396 | 0.525 | 0.620 |
|        | **Sonnet Full**      | 0.205 | 0.418 | 0.528 | 0.646 |
|        | **Sonnet Filtered**  | 0.199 | 0.426 | 0.541 | 0.643 |

### nDCG Metrics

| Domain | Condition            | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10   |
| ------ | -------------------- | ------ | ------ | ------ | --------- |
| All    | **Mixtral Full**     | 0.377  | 0.366  | 0.405  | 0.454     |
|        | **Mixtral Filtered** | 0.384  | 0.370  | 0.407  | 0.458     |
|        | **Sonnet Full**      | 0.441  | 0.407  | 0.449  | **0.501** |
|        | **Sonnet Filtered**  | 0.440  | 0.405  | 0.444  | 0.495     |
| ClapNQ | **Mixtral Full**     | 0.500  | 0.475  | 0.514  | 0.574     |
|        | **Mixtral Filtered** | 0.481  | 0.464  | 0.493  | 0.562     |
|        | **Sonnet Full**      | 0.548  | 0.493  | 0.539  | 0.604     |
|        | **Sonnet Filtered**  | 0.558  | 0.501  | 0.547  | 0.609     |
| Cloud  | **Mixtral Full**     | 0.287  | 0.273  | 0.300  | 0.344     |
|        | **Mixtral Filtered** | 0.298  | 0.290  | 0.328  | 0.365     |
|        | **Sonnet Full**      | 0.362  | 0.334  | 0.375  | 0.414     |
|        | **Sonnet Filtered**  | 0.346  | 0.321  | 0.353  | 0.392     |
| FiQA   | **Mixtral Full**     | 0.322  | 0.296  | 0.332  | 0.385     |
|        | **Mixtral Filtered** | 0.300  | 0.292  | 0.318  | 0.374     |
|        | **Sonnet Full**      | 0.411  | 0.355  | 0.391  | 0.443     |
|        | **Sonnet Filtered**  | 0.394  | 0.343  | 0.376  | 0.436     |
| Govt   | **Mixtral Full**     | 0.383  | 0.404  | 0.455  | 0.496     |
|        | **Mixtral Filtered** | 0.438  | 0.417  | 0.470  | 0.511     |
|        | **Sonnet Full**      | 0.433  | 0.432  | 0.478  | 0.528     |
|        | **Sonnet Filtered**  | 0.448  | 0.439  | 0.484  | 0.528     |

---

## Context Filtering Statistics

**Note**: Statistics are for the **777 evaluable tasks** (tasks with relevance judgments).
The full pipeline processed 842 tasks, but 65 unanswerable queries have no qrels.

### Overview

| Metric                              | Sonnet Filtered | Mixtral Filtered |
| ----------------------------------- | --------------- | ---------------- |
| **Total evaluable tasks**           | 777             | 777              |
| **Turn 1 (no history)**             | 102             | 102              |
| **Multi-turn queries (Turn > 1)**   | 675             | 675              |
| **Average history turns available** | 3.98            | 3.98             |
| **Average turns SELECTED**          | 1.99            | 1.95             |
| **Average turns FILTERED OUT**      | 1.99            | 2.03             |
| **Average retention ratio**         | **59.6%**       | **59.1%**        |
| **Average filtering ratio**         | **40.4%**       | **40.9%**        |

The slight differences between Sonnet and Mixtral occur because filtering is based on semantic similarity between history turns and the LLM-generated response—different LLMs produce different responses, leading to different similarity scores.

## Key Findings

### Filtering Applied

| Metric                      | Sonnet Filtered | Mixtral Filtered |
| --------------------------- | --------------- | ---------------- |
| Some context filtered       | 450 (66.7%)     | 442 (65.5%)      |
| All context kept            | 225 (33.3%)     | 233 (34.5%)      |
| All context filtered        | 0 (0%)          | 0 (0%)           |

Queries that kept all context are mostly Turn 2 queries with only 1 history turn, which is always included per `include_last_turn=True`. No queries had all context filtered because the last turn is always kept.

### Retention Rate Distribution

| Retention           | Sonnet Cases | Sonnet % | Mixtral Cases | Mixtral % |
| ------------------- | ------------ | -------- | ------------- | --------- |
| 100% (no filtering) | 225          | 33.3%    | 233           | 34.5%     |
| 75-99%              | 35           | 5.2%     | 30            | 4.4%      |
| 50-75%              | 147          | 21.8%    | 133           | 19.7%     |
| 25-50%              | 154          | 22.8%    | 155           | 23.0%     |
| 0-25%               | 114          | 16.9%    | 124           | 18.4%     |

### Filtering by Conversation Depth

As conversations get longer, more context gets filtered:

| History Depth | Cases | Sonnet Avg Sel | Sonnet Ret | Mixtral Avg Sel | Mixtral Ret |
| ------------- | ----- | -------------- | ---------- | --------------- | ----------- |
| 1 turn        | 106   | 1.0            | 100.0%     | 1.0             | 100.0%      |
| 2 turns       | 103   | 1.4            | 68.4%      | 1.4             | 69.9%       |
| 3 turns       | 99    | 1.7            | 56.2%      | 1.7             | 55.9%       |
| 4 turns       | 96    | 2.2            | 54.7%      | 2.1             | 52.1%       |
| 5 turns       | 87    | 2.6            | 52.9%      | 2.6             | 52.4%       |
| 6 turns       | 80    | 2.5            | 42.1%      | 2.5             | 42.3%       |
| 7 turns       | 65    | 2.7            | 38.7%      | 2.6             | 37.4%       |
| 8 turns       | 32    | 2.5            | 31.6%      | 2.3             | 28.5%       |

**Key insight**: On average, only ~2 turns are selected regardless of conversation length, suggesting the semantic similarity filtering is quite aggressive in keeping only the most relevant context. Both LLMs show similar filtering patterns, with Mixtral being slightly more aggressive (lower retention at deeper depths).

## Source

Experiment code:

- `scripts/ideas/retrieval_tasks/baseline_rewrite_with_mixtral/` (Mixtral Full)
- `scripts/ideas/retrieval_tasks/targeted_rewrite_with_mixtral/` (Mixtral Filtered)
- `scripts/ideas/retrieval_tasks/baseline_rewrite_with_sonnet/` (Sonnet Full)
- `scripts/ideas/retrieval_tasks/targeted_rewrite_with_sonnet/` (Sonnet Filtered)
