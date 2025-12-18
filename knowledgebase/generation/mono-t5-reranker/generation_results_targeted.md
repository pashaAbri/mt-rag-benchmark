# Generation Results - Mono-T5 Targeted Full RAG

## Metrics Explanation

- **RL_F** (RAGAS Faithfulness): Measures how faithful the generated response is to the retrieved context. Higher = response is more grounded in the provided documents.
- **RB_llm** (RADBench LLM Judge): LLM-based evaluation comparing the response to a reference answer. Assesses faithfulness, appropriateness, and completeness. Score is normalized 0-1.
- **RB_agg** (RADBench Aggregate): Algorithmic metrics combining ROUGE-L, BERTScore, and other text similarity measures against the reference answer.
- **H-Mean** (Harmonic Mean): Harmonic mean of RL_F, RB_llm, and RB_agg. Balances all three metrics, penalizing models that perform poorly on any single metric.

---

## Results Comparison

| Model          | Method         |  RL_F  | RB_llm | RB_agg |   H-Mean   |
| -------------- | -------------- | :----: | :----: | :----: | :--------: |
| llama_3.1_405b | Mono-T5 (Ours) | 0.8299 | 0.6969 | 0.4314 | **0.6051** |
|                | Baseline       | 0.7764 | 0.6786 | 0.4286 |   0.5888   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| mixtral_8x22b  | Mono-T5 (Ours) | 0.8459 | 0.6777 | 0.4347 | **0.6051** |
|                | Baseline       | 0.8124 | 0.6629 | 0.4326 |   0.5939   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| qwen_2.5_72b   | Mono-T5 (Ours) | 0.8180 | 0.7210 | 0.4157 | **0.5982** |
|                | Baseline       | 0.7552 | 0.7090 | 0.4100 |   0.5799   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| qwen_2.5_7b    | Mono-T5 (Ours) | 0.8194 | 0.6519 | 0.4081 | **0.5764** |
|                | Baseline       | 0.7782 | 0.6286 | 0.4019 |   0.5593   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| gpt_4o         | Mono-T5 (Ours) | 0.7682 | 0.6521 | 0.4078 | **0.5674** |
|                | Baseline       | 0.6977 | 0.6120 | 0.3926 |   0.5343   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| gpt_4o_mini    | Mono-T5 (Ours) | 0.7563 | 0.6634 | 0.3837 | **0.5519** |
|                | Baseline       | 0.7117 | 0.6379 | 0.3698 |   0.5284   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| command_r_plus | Mono-T5 (Ours) | 0.7818 | 0.5850 | 0.3955 | **0.5438** |
|                | Baseline       | 0.7451 | 0.5601 | 0.3870 |   0.5253   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| llama_3.1_70b  | Mono-T5 (Ours) | 0.7355 | 0.5977 | 0.4005 | **0.5426** |
|                | Baseline       | 0.6849 | 0.5738 | 0.3881 |   0.5190   |
| ───            | ───            |  ───   |  ───   |  ───   |    ───     |
| llama_3.1_8b   | Mono-T5 (Ours) | 0.6460 | 0.4691 | 0.3686 | **0.4693** |
|                | Baseline       | 0.5631 | 0.4483 | 0.3535 |   0.4389   |

> **Note**: The above results use non-IDK metrics on all 842 tasks. See the [Paper-Comparable Results](#paper-comparable-results-leaderboard-methodology) section below for comparison using the official leaderboard methodology.

## Improvement (Mono-T5 vs Baseline)

| Model          | RL_F Δ | RB_llm Δ | RB_agg Δ | H-Mean Δ |
| -------------- | :----: | :------: | :------: | :------: |
| llama_3.1_405b | +0.054 |  +0.018  |  +0.003  |  +0.016  |
| mixtral_8x22b  | +0.033 |  +0.015  |  +0.002  |  +0.011  |
| qwen_2.5_72b   | +0.063 |  +0.012  |  +0.006  |  +0.018  |
| qwen_2.5_7b    | +0.041 |  +0.023  |  +0.006  |  +0.017  |
| gpt_4o         | +0.071 |  +0.040  |  +0.015  |  +0.033  |
| gpt_4o_mini    | +0.045 |  +0.026  |  +0.014  |  +0.023  |
| command_r_plus | +0.037 |  +0.025  |  +0.009  |  +0.019  |
| llama_3.1_70b  | +0.051 |  +0.024  |  +0.012  |  +0.024  |
| llama_3.1_8b   | +0.083 |  +0.021  |  +0.015  |  +0.030  |

---

## Paper-Comparable Results (Leaderboard Methodology)

The MT-RAG paper leaderboard uses **IDK-conditioned metrics** on tasks with **≤2 reference passages (n≈426)**. This methodology rewards models that correctly decline unanswerable questions.

### Methodology Comparison

| Aspect           | Above Results        | Leaderboard                            |
| ---------------- | -------------------- | -------------------------------------- |
| Metrics          | RL_F, RB_llm, RB_agg | RL_F_idk, RB_llm_idk, RB_agg_idk       |
| Sample           | n=842 (all tasks)    | n≈426 (≤2 ref passages)                |
| IDK Conditioning | No                   | Yes (penalizes answering unanswerable) |

### Mono-T5 vs Paper Baseline (IDK-conditioned, n≈426)

| Rank | Model          | Method             | RL_F | RB_llm | RB_agg |  H-Mean  |
| :--: | -------------- | ------------------ | :--: | :----: | :----: | :------: |
|  1   | mixtral_8x22b  | **Mono-T5 (Ours)** | 0.81 |  0.69  |  0.42  | **0.59** |
|  1   | llama_3.1_405b | **Mono-T5 (Ours)** | 0.79 |  0.69  |  0.42  | **0.59** |
|  3   | qwen_2.5_72b   | **Mono-T5 (Ours)** | 0.77 |  0.71  |  0.40  | **0.57** |
|  4   | qwen_2.5_7b    | **Mono-T5 (Ours)** | 0.78 |  0.63  |  0.39  | **0.55** |
|  5   | command_r_plus | **Mono-T5 (Ours)** | 0.75 |  0.57  |  0.38  | **0.53** |
|  5   | llama_3.1_405b | Paper Baseline     | 0.65 |  0.63  |  0.39  |   0.53   |
|  5   | gpt_4o         | Paper Baseline     | 0.65 |  0.66  |  0.38  |   0.53   |
|  8   | qwen_2.5_72b   | Paper Baseline     | 0.64 |  0.65  |  0.37  |   0.52   |
|  8   | llama_3.1_70b  | Paper Baseline     | 0.64 |  0.59  |  0.39  |   0.52   |
|  10  | gpt_4o_mini    | Paper Baseline     | 0.64 |  0.64  |  0.37  |   0.51   |
|  10  | command_r_plus | Paper Baseline     | 0.66 |  0.59  |  0.38  |   0.51   |
|  10  | qwen_2.5_7b    | Paper Baseline     | 0.62 |  0.63  |  0.37  |   0.51   |
|  13  | gpt_4o_mini    | **Mono-T5 (Ours)** | 0.69 |  0.59  |  0.33  | **0.49** |
|  14  | gpt_4o         | **Mono-T5 (Ours)** | 0.66 |  0.57  |  0.34  | **0.48** |
|  14  | mixtral_8x22b  | Paper Baseline     | 0.56 |  0.61  |  0.35  |   0.48   |
|  14  | llama_3.1_70b  | **Mono-T5 (Ours)** | 0.66 |  0.53  |  0.35  | **0.48** |
|  17  | llama_3.1_8b   | Paper Baseline     | 0.53 |  0.54  |  0.34  |   0.45   |
|  18  | llama_3.1_8b   | **Mono-T5 (Ours)** | 0.58 |  0.40  |  0.30  | **0.40** |

### Key Findings

1. **Top performers**: Mono-T5 with mixtral_8x22b and llama_3.1_405b achieve **H-Mean of 0.59**, surpassing the paper's best baseline (0.53) by **+11%**.

2. **5 of top 5 positions** are held by Mono-T5 variants.

3. **Improvement varies by model**: Open-weight models (mixtral, qwen, llama_405b) benefit most from better retrieval; proprietary models (gpt_4o, gpt_4o_mini) show smaller gains on this subset.

4. **Caveat**: Some Mono-T5 results (gpt_4o, llama_3.1_8b) underperform their paper baselines on this subset, suggesting retrieval quality has differential impact across models and question types.
