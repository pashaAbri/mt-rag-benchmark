# Aggressive Rewrite vs. Baseline Rewrite: Performance Comparison

## Overview

We evaluated the "Aggressive Rewrite" strategy, which prioritizes entity resolution and context injection over conversational naturalness, against the standard Baseline Rewrite strategy.

**Metric:** nDCG@5 (Normalized Discounted Cumulative Gain at rank 5) using ELSER retrieval.

## Results Table

| Domain     | Baseline nDCG@5 | Aggressive nDCG@5 | Absolute Δ | Relative Δ |
| :--------- | :-------------- | :---------------- | :--------- | :--------- |
| **ClapNQ** | 0.5135          | **0.5627**        | +0.0492    | **+9.58%** |
| **Govt**   | 0.4540          | **0.4857**        | +0.0317    | **+6.98%** |
| **FiQA**   | 0.3779          | 0.3657            | -0.0122    | -3.23%     |
| **Cloud**  | 0.3940          | 0.3278            | -0.0662    | -16.80%    |

## Analysis

### 1. Wins: ClapNQ and Govt

The aggressive strategy achieved **significant gains** in ClapNQ (+9.6%) and Govt (+7.0%).

- **Reason**: These domains often feature multi-turn conversations about entities (people, events, laws) where pronouns ("he", "it", "the bill") are common. The aggressive entity resolution successfully disambiguated these queries.
- **Example Success**: Resolving "the movement" to "Quit India Movement" or "he" to "Doctor Strange".

### 2. Losses: Cloud and FiQA

The strategy caused regressions in Cloud (-16.8%) and FiQA (-3.2%).

- **Reason**: These domains (Technical/Financial) often require **precision**.
- **Hypothesis**: The "context injection" rule likely led to **query drift** or **keyword stuffing** in technical queries.
  - _Example Failure Pattern_: A query like "What is an IAM credential?" becoming a 20-word string of related keywords might match broad documentation rather than the specific definition page.
  - _Hallucination_: The model might have inferred incorrect specific contexts (e.g., adding "in IBM Cloud" to a general Kubernetes question when the answer might be in a different context or valid generally).

## Recommendation

- **Adopt Aggressive Rewrite for**: General Knowledge (ClapNQ) and Policy/Government (Govt) tasks.
- **Refine Strategy for**: Technical (Cloud) and Financial (FiQA) tasks. The "minimalism" of the baseline might be a feature, not a bug, for precision-heavy domains.
- **Hybrid Approach**: A routing strategy that uses Aggressive Rewrite for ambiguous/short queries and Baseline for precise/technical queries could yield the best of both worlds.
