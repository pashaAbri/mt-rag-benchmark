# Lambda Parameter Experiments Analysis

This document analyzes the impact of the lambda (λ) parameter on MMR Clustering performance. Lambda controls the diversity vs. relevance trade-off in MMR selection:

- **Higher lambda (0.85, 0.9)**: More relevance, less diversity - prioritizes sentences most similar to the query
- **Lower lambda (0.3, 0.5)**: More diversity, less relevance - prioritizes diverse sentence selection

**Dataset**: 777 tasks (Answerable + Partial only) - filtered to match baselines for fair comparison.

**Retriever**: ELSER (Elastic Learned Sparse Encoder)

**Configuration**: k=10 (fixed), lambda values tested: 0.3, 0.5, 0.7 (baseline), 0.85, 0.9

## Overall Performance Summary (All Domains)

| Strategy                  | R@5        | R@10       | NDCG@5     | NDCG@10    | Total Tasks |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| Last Turn                 | 0.4394     | 0.5445     | 0.4054     | 0.4511     | 777         |
| Query Rewrite             | **0.4761** | **0.6078** | **0.4378** | **0.4953** | 777         |
| Questions                 | 0.2694     | 0.3570     | 0.2368     | 0.2740     | 777         |
| MMR Cluster (λ=0.3, k10)  | 0.4574     | 0.5854     | 0.4180     | 0.4727     | 777         |
| MMR Cluster (λ=0.5, k10)  | 0.4562     | 0.5678     | 0.4137     | 0.4615     | 777         |
| MMR Cluster (λ=0.7, k10)  | 0.4648     | 0.5781     | 0.4226     | 0.4710     | 777         |
| MMR Cluster (λ=0.85, k10) | 0.4513     | 0.5794     | 0.4154     | 0.4702     | 777         |
| MMR Cluster (λ=0.9, k10)  | 0.4555     | 0.5673     | 0.4156     | 0.4642     | 777         |

## Domain-by-Domain Lambda Analysis

### ClapNQ Domain

| Strategy                  | R@5        | R@10       | NDCG@5     | NDCG@10    | Total Tasks |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| Last Turn                 | 0.5113     | 0.6303     | 0.4749     | 0.5270     | 208         |
| Query Rewrite             | 0.5516     | 0.7005     | 0.5135     | 0.5781     | 208         |
| Questions                 | 0.3016     | 0.4087     | 0.2692     | 0.3153     | 208         |
| MMR Cluster (λ=0.3, k10)  | 0.5607     | **0.7167** | 0.5178     | **0.5852** | 208         |
| MMR Cluster (λ=0.5, k10)  | **0.5695** | 0.6926     | 0.5154     | 0.5699     | 208         |
| MMR Cluster (λ=0.7, k10)  | 0.5555     | 0.6994     | 0.5087     | 0.5704     | 208         |
| MMR Cluster (λ=0.85, k10) | 0.5398     | 0.6945     | 0.5021     | 0.5698     | 208         |
| MMR Cluster (λ=0.9, k10)  | 0.5651     | 0.6978     | **0.5227** | 0.5805     | 208         |

### Cloud Domain

| Strategy                  | R@5        | R@10       | NDCG@5     | NDCG@10    | Total Tasks |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| Last Turn                 | 0.4201     | 0.5037     | 0.3894     | 0.4273     | 188         |
| Query Rewrite             | **0.4297** | **0.5280** | **0.3940** | **0.4377** | 188         |
| Questions                 | 0.2180     | 0.3037     | 0.1861     | 0.2220     | 188         |
| MMR Cluster (λ=0.3, k10)  | 0.4258     | 0.5226     | 0.3838     | 0.4261     | 188         |
| MMR Cluster (λ=0.5, k10)  | 0.4052     | 0.5188     | 0.3694     | 0.4164     | 188         |
| MMR Cluster (λ=0.7, k10)  | 0.4092     | 0.5031     | 0.3658     | 0.4059     | 188         |
| MMR Cluster (λ=0.85, k10) | 0.4015     | 0.4982     | 0.3669     | 0.4085     | 188         |
| MMR Cluster (λ=0.9, k10)  | 0.4023     | 0.4859     | 0.3705     | 0.4063     | 188         |

**Best Lambda**: **λ=0.3** consistently best across all metrics

### FiQA Domain

| Strategy                  | R@5        | R@10       | NDCG@5     | NDCG@10    | Total Tasks |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| Last Turn                 | 0.3705     | 0.4719     | 0.3477     | 0.3909     | 180         |
| Query Rewrite             | **0.4017** | **0.5357** | **0.3779** | **0.4355** | 180         |
| Questions                 | 0.1913     | 0.2494     | 0.1818     | 0.2071     | 180         |
| MMR Cluster (λ=0.3, k10)  | 0.3693     | 0.5036     | 0.3437     | 0.4008     | 180         |
| MMR Cluster (λ=0.5, k10)  | 0.3642     | 0.4524     | 0.3452     | 0.3831     | 180         |
| MMR Cluster (λ=0.7, k10)  | 0.3730     | 0.4654     | 0.3514     | 0.3907     | 180         |
| MMR Cluster (λ=0.85, k10) | 0.3810     | 0.5108     | 0.3558     | 0.4113     | 180         |
| MMR Cluster (λ=0.9, k10)  | 0.3559     | 0.4672     | 0.3326     | 0.3819     | 180         |

**Best Lambda**: **λ=0.85** consistently best across all metrics

### Govt Domain

| Strategy                  | R@5        | R@10       | NDCG@5     | NDCG@10    | Total Tasks |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| Last Turn                 | 0.4449     | 0.5588     | 0.4001     | 0.4486     | 201         |
| Query Rewrite             | **0.5082** | **0.6510** | **0.4540** | **0.5169** | 201         |
| Questions                 | 0.3540     | 0.4498     | 0.3000     | 0.3397     | 201         |
| MMR Cluster (λ=0.3, k10)  | 0.4589     | 0.5817     | 0.4133     | 0.4644     | 201         |
| MMR Cluster (λ=0.5, k10)  | 0.4689     | 0.5880     | 0.4115     | 0.4618     | 201         |
| MMR Cluster (λ=0.7, k10)  | 0.5052     | 0.6237     | 0.4505     | 0.5008     | 201         |
| MMR Cluster (λ=0.85, k10) | 0.4691     | 0.5976     | 0.4243     | 0.4775     | 201         |
| MMR Cluster (λ=0.9, k10)  | 0.4809     | 0.5979     | 0.4212     | 0.4718     | 201         |

**Best Lambda**: **λ=0.7** consistently best across all metrics

## Lambda Impact Summary

### Overall Performance Changes vs. Baseline (λ=0.7)

| Lambda | R@5 Change       | R@10 Change      | NDCG@5 Change    | NDCG@10 Change   |
| ------ | ---------------- | ---------------- | ---------------- | ---------------- |
| 0.3    | -0.0074 (-1.60%) | +0.0074 (+1.27%) | -0.0046 (-1.09%) | +0.0018 (+0.38%) |
| 0.5    | -0.0086 (-1.85%) | -0.0102 (-1.77%) | -0.0089 (-2.10%) | -0.0094 (-2.01%) |
| 0.85   | -0.0135 (-2.91%) | +0.0013 (+0.22%) | -0.0073 (-1.72%) | -0.0008 (-0.17%) |
| 0.9    | -0.0093 (-2.01%) | -0.0108 (-1.87%) | -0.0070 (-1.67%) | -0.0068 (-1.43%) |
