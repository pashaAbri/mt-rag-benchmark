# Comprehensive Comparison: Context Filtering vs Full History

This document provides a complete comparison of all four experimental conditions, isolating the effects of **LLM quality** and **context filtering**.

## Experimental Conditions

| Condition            | LLM           | Context Strategy | Description                   |
| -------------------- | ------------- | ---------------- | ----------------------------- |
| **Mixtral Full**     | Mixtral 8x7B  | Full history     | Paper baseline                |
| **Mixtral Filtered** | Mixtral 8x7B  | Similarity ≥ 0.3 | Targeted rewrite with Mixtral |
| **Sonnet Full**      | Claude Sonnet | Full history     | Fair baseline for Sonnet      |
| **Sonnet Filtered**  | Claude Sonnet | Similarity ≥ 0.3 | Targeted rewrite with Sonnet  |

---

## Results

### Recall Metrics

| Domain | Condition            | R@1   | R@3   | R@5   | R@10  |
| ------ | -------------------- | ----- | ----- | ----- | ----- |
| All    | **Mixtral Full**     | 0.187 | 0.372 | 0.476 | 0.608 |
|        | **Mixtral Filtered** | 0.163 | 0.348 | 0.444 | 0.559 |
|        | **Sonnet Full**      | 0.190 | 0.377 | 0.484 | 0.603 |
|        | **Sonnet Filtered**  | 0.186 | 0.377 | 0.479 | 0.597 |
| ClapNQ | **Mixtral Full**     | 0.209 | 0.424 | 0.552 | 0.701 |
|        | **Mixtral Filtered** | 0.183 | 0.434 | 0.532 | 0.685 |
|        | **Sonnet Full**      | 0.220 | 0.445 | 0.580 | 0.726 |
|        | **Sonnet Filtered**  | 0.222 | 0.454 | 0.588 | 0.732 |
| Cloud  | **Mixtral Full**     | 0.179 | 0.353 | 0.430 | 0.528 |
|        | **Mixtral Filtered** | 0.137 | 0.283 | 0.373 | 0.460 |
|        | **Sonnet Full**      | 0.160 | 0.321 | 0.414 | 0.505 |
|        | **Sonnet Filtered**  | 0.155 | 0.312 | 0.384 | 0.479 |
| FiQA   | **Mixtral Full**     | 0.163 | 0.310 | 0.402 | 0.536 |
|        | **Mixtral Filtered** | 0.128 | 0.280 | 0.347 | 0.478 |
|        | **Sonnet Full**      | 0.175 | 0.324 | 0.414 | 0.532 |
|        | **Sonnet Filtered**  | 0.169 | 0.317 | 0.400 | 0.537 |
| Govt   | **Mixtral Full**     | 0.194 | 0.392 | 0.508 | 0.651 |
|        | **Mixtral Filtered** | 0.202 | 0.396 | 0.525 | 0.620 |
|        | **Sonnet Full**      | 0.205 | 0.418 | 0.528 | 0.646 |
|        | **Sonnet Filtered**  | 0.199 | 0.426 | 0.541 | 0.643 |

### nDCG Metrics

| Domain | Condition            | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10   |
| ------ | -------------------- | ------ | ------ | ------ | --------- |
| All    | **Mixtral Full**     | 0.426  | 0.397  | 0.435  | 0.492     |
|        | **Mixtral Filtered** | 0.379  | 0.366  | 0.402  | 0.453     |
|        | **Sonnet Full**      | 0.438  | 0.404  | 0.446  | **0.497** |
|        | **Sonnet Filtered**  | 0.436  | 0.401  | 0.440  | 0.491     |
| ClapNQ | **Mixtral Full**     | 0.524  | 0.470  | 0.513  | 0.578     |
|        | **Mixtral Filtered** | 0.481  | 0.464  | 0.493  | 0.562     |
|        | **Sonnet Full**      | 0.548  | 0.493  | 0.539  | 0.604     |
|        | **Sonnet Filtered**  | 0.558  | 0.501  | 0.547  | 0.609     |
| Cloud  | **Mixtral Full**     | 0.378  | 0.365  | 0.394  | 0.438     |
|        | **Mixtral Filtered** | 0.298  | 0.290  | 0.328  | 0.365     |
|        | **Sonnet Full**      | 0.362  | 0.334  | 0.375  | 0.414     |
|        | **Sonnet Filtered**  | 0.346  | 0.321  | 0.353  | 0.392     |
| FiQA   | **Mixtral Full**     | 0.389  | 0.344  | 0.378  | 0.436     |
|        | **Mixtral Filtered** | 0.300  | 0.292  | 0.318  | 0.374     |
|        | **Sonnet Full**      | 0.411  | 0.355  | 0.391  | 0.443     |
|        | **Sonnet Filtered**  | 0.394  | 0.343  | 0.376  | 0.436     |
| Govt   | **Mixtral Full**     | 0.413  | 0.407  | 0.454  | 0.517     |
|        | **Mixtral Filtered** | 0.438  | 0.417  | 0.470  | 0.511     |
|        | **Sonnet Full**      | 0.433  | 0.432  | 0.478  | 0.528     |
|        | **Sonnet Filtered**  | 0.448  | 0.439  | 0.484  | 0.528     |
---

## Source

Experiment code:

- `scripts/ideas/retrieval_tasks/targeted_rewrite_with_mixtral/` (Mixtral Filtered)
- `scripts/ideas/retrieval_tasks/targeted_rewrite_with_sonnet/` (Sonnet Filtered)
- `scripts/ideas/retrieval_tasks/baseline_rewrite_with_sonnet/` (Sonnet Full)
- `scripts/baselines/retrieval_scripts/elser/` (Mixtral Full / Paper Baseline)
