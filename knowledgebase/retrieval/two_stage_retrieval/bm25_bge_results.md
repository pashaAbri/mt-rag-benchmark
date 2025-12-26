# Two-Stage BM25 + BGE: Comprehensive Results

## Method Overview

Two-stage retrieval pipeline:

1. **Stage 1**: BM25 lexical retrieval (local, PyTerrier) retrieves top-k candidates
2. **Stage 2**: BGE neural reranking (local, sentence-transformers) reranks to top 10

### Configuration

| Parameter           | Value                   |
| :------------------ | :---------------------- |
| BM25 Implementation | PyTerrier               |
| BGE Model           | `BAAI/bge-base-en-v1.5` |
| BM25 Candidates     | 500                     |
| Final Top-k         | 10                      |
| Query Type          | Rewrite                 |

---

## Comprehensive Results

| Domain     | Method               | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | R@1   | R@3   | R@5   | R@10  |
| :--------- | :------------------- | :----- | :----- | :----- | :------ | :---- | :---- | :---- | :---- |
| **ClapNQ** | BM25@500+BGE         | 0.486  | 0.418  | 0.450  | 0.505   | 0.184 | 0.379 | 0.479 | 0.610 |
|            | Baseline BGE (std)   | 0.471  | 0.413  | 0.438  | 0.498   | 0.174 | 0.376 | 0.462 | 0.606 |
|            | Baseline ELSER (std) | 0.524  | 0.470  | 0.514  | 0.578   | 0.209 | 0.424 | 0.552 | 0.701 |
|            |                      |        |        |        |         |       |       |       |       |
| **Cloud**  | BM25@500+BGE         | 0.309  | 0.287  | 0.312  | 0.355   | 0.152 | 0.278 | 0.338 | 0.431 |
|            | Baseline BGE (std)   | 0.293  | 0.276  | 0.304  | 0.342   | 0.148 | 0.271 | 0.338 | 0.423 |
|            | Baseline ELSER (std) | 0.378  | 0.365  | 0.394  | 0.438   | 0.179 | 0.353 | 0.430 | 0.528 |
|            |                      |        |        |        |         |       |       |       |       |
| **FiQA**   | BM25@500+BGE         | 0.300  | 0.262  | 0.285  | 0.325   | 0.124 | 0.232 | 0.298 | 0.390 |
|            | Baseline BGE (std)   | 0.311  | 0.274  | 0.294  | 0.341   | 0.130 | 0.249 | 0.308 | 0.418 |
|            | Baseline ELSER (std) | 0.389  | 0.344  | 0.378  | 0.436   | 0.163 | 0.310 | 0.402 | 0.536 |
|            |                      |        |        |        |         |       |       |       |       |
| **Govt**   | BM25@500+BGE         | 0.368  | 0.339  | 0.378  | 0.429   | 0.166 | 0.320 | 0.412 | 0.532 |
|            | Baseline BGE (std)   | 0.343  | 0.329  | 0.368  | 0.420   | 0.157 | 0.313 | 0.404 | 0.528 |
|            | Baseline ELSER (std) | 0.413  | 0.407  | 0.454  | 0.517   | 0.194 | 0.392 | 0.508 | 0.651 |

---

## Summary: nDCG@10 Comparison

| Domain     | BM25@500+BGE | Baseline BGE | Baseline ELSER | Gap vs BGE | Gap vs ELSER |
| :--------- | :----------- | :----------- | :------------- | :--------- | :----------- |
| **ClapNQ** | 0.505        | 0.498        | 0.578          | **+1.4%**  | -12.6%       |
| **Cloud**  | 0.355        | 0.342        | 0.438          | **+3.8%**  | -18.9%       |
| **FiQA**   | 0.325        | 0.341        | 0.436          | -4.7%      | -25.5%       |
| **Govt**   | 0.429        | 0.420        | 0.517          | **+2.1%**  | -17.0%       |

---

## Key Findings

1. **BM25@500+BGE beats Baseline BGE** on 3/4 domains (ClapNQ, Cloud, Govt)

2. **BM25@500+BGE vs Baseline ELSER**:

   - Underperforms ELSER on all domains
   - Average gap: ~18.5%
   - ELSER remains the best single-stage retriever

3. **Fully local pipeline**: No cloud dependencies required

---

## Trade-offs

| Aspect             | BM25@500+BGE                    | Baseline BGE                |
| :----------------- | :------------------------------ | :-------------------------- |
| **Performance**    | Slightly better (+1-4%)         | Slightly worse on 3 domains |
| **Infrastructure** | Fully local                     | Fully local                 |
| **Latency**        | BM25 fast + BGE encode 500 docs | Single dense retrieval      |
