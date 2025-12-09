# MMR Clustering Statistics: Filtering and Selection Analysis

## Experiment Configuration

- **k (max selected sentences)**: 10
- **Lambda (MMR diversity parameter)**: 0.7
- **Clustering method**: BGE embeddings with hierarchical clustering
- **Retriever**: ELSER (for final retrieval after rewriting)

## Overview

This document provides comprehensive statistics on how MMR Clustering filters and selects context from conversation history. The clustering process extracts sentences, groups them into clusters, selects representative sentences, and then applies MMR to choose the final set for query rewriting.

**Total turns analyzed**: 732 (excluding Turn 1, which has no context to cluster)

## Key Findings

### 1. Clustering as a Filtering Mechanism

The MMR Clustering process acts as a **progressive filtering mechanism**:

1. **Extract** sentences from conversation history
2. **Cluster** them into semantic groups
3. **Select representatives** from each cluster
4. **Apply MMR** to choose diverse final sentences (capped at k=10)

**Overall Reduction**: On average, the system reduces from **26.8 extracted sentences** to **8.9 selected sentences** (66.8% reduction).

### 2. Number of Clusters Increases with Turn

| Turn | Mean Clusters | Median | Range | Notes                                |
| ---- | ------------- | ------ | ----- | ------------------------------------ |
| 2    | 2.17          | 2      | 1-4   | Early conversations, limited context |
| 3    | 3.11          | 3      | 2-5   | Growing context diversity            |
| 4    | 3.99          | 4      | 3-6   | Moderate diversity                   |
| 5    | 4.72          | 5      | 3-7   | High diversity                       |
| 6    | 5.29          | 5      | 4-7   | Very diverse topics                  |
| 7    | 5.81          | 6      | 4-7   | Maximum diversity                    |
| 8    | 6.26          | 6      | 4-7   | Long conversations                   |
| 9+   | 6.5-7.0       | 7      | 5-7   | Very long conversations              |

**Insight**: As conversations get longer, more distinct topics emerge, leading to more clusters. The system reaches the maximum of 7 clusters around Turn 7-8.

### 3. Cluster Count Distribution

| Number of Clusters | Frequency | Percentage |
| ------------------ | --------- | ---------- |
| 1                  | 2         | 0.3%       |
| 2                  | 108       | 14.8%      |
| 3                  | 112       | 15.3%      |
| 4                  | 143       | 19.5%      |
| **5**              | **161**   | **22.0%**  |
| 6                  | 123       | 16.8%      |
| 7                  | 83        | 11.3%      |

**Most common**: 4-5 clusters (41.5% of all cases)

### 4. Progressive Filtering by Turn

| Turn | Extracted | Representatives | Selected (MMR) | Reduction |
| ---- | --------- | --------------- | -------------- | --------- |
| 2    | 6.3       | 5.1             | 5.0            | 20.4%     |
| 3    | 12.9      | 8.4             | 7.9            | 38.6%     |
| 4    | 20.1      | 11.2            | 9.5            | 52.9%     |
| 5    | 27.0      | 13.6            | 10.0           | 63.0%     |
| 6    | 33.5      | 15.4            | 10.0           | 70.2%     |
| 7    | 40.5      | 17.1            | 10.0           | 75.3%     |
| 8    | 47.2      | 18.5            | 10.0           | 78.8%     |
| 9    | 52.8      | 19.4            | 10.0           | 81.0%     |
| 10   | 63.8      | 19.8            | 10.0           | 84.3%     |
| 11   | 101.0     | 21.0            | 10.0           | 90.1%     |
| 12   | 113.0     | 21.0            | 10.0           | 91.2%     |

**Key Observations**:

1. **Extracted sentences grow linearly** with turn number (more history = more sentences)
2. **Representative sentences grow** but at a slower rate (clustering helps reduce redundancy)
3. **Selected sentences cap at k=10** - After Turn 5, almost all cases select exactly 10 sentences
4. **Filtering becomes more aggressive** in later turns - up to 91% reduction at Turn 12

## Statistics by Domain

### Cloud Domain (179 cases)

- **Mean clusters**: 4.59
- **Mean extracted**: 29.5 sentences
- **Mean selected**: 9.1 sentences
- **Reduction**: 69.2%

### ClapNQ Domain (195 cases)

- **Mean clusters**: 4.07 (lowest)
- **Mean extracted**: 22.1 sentences (lowest)
- **Mean selected**: 8.6 sentences
- **Reduction**: 61.0% (lowest reduction)

### FiQA Domain (172 cases)

- **Mean clusters**: 4.68
- **Mean extracted**: 28.7 sentences
- **Mean selected**: 9.1 sentences
- **Reduction**: 68.3%

### Govt Domain (186 cases)

- **Mean clusters**: 4.46
- **Mean extracted**: 27.3 sentences
- **Mean selected**: 8.7 sentences
- **Reduction**: 68.2%

**Domain Differences**: ClapNQ has fewer clusters and less aggressive filtering, suggesting more coherent, topic-focused conversations. Cloud has the most aggressive filtering (highest reduction rate).
