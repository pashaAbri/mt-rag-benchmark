# Full RAG Setting - Real-World Pipeline

This scenario represents the **real-world RAG pipeline** with end-to-end retrieval and generation.

## Overview

- **Setting**: Full RAG (◦)
- **Tasks**: 842 (all tasks from benchmark)
- **Passages**: Top 5 from Elser retrieval with query rewrite
- **Purpose**: Test complete RAG system (most realistic)

## Retrieval Strategy

- **Pure retrieval** using Elser sparse retrieval
- **Query rewrite** to handle non-standalone questions
- **Top 5 passages** returned
- **No guarantees** that relevant passages are retrieved

This is the **standard RAG setting** used in production systems.

## Usage

### Run Generation

```bash
cd scripts/baselines/generation_scripts/full_rag
./run_full_rag.sh
```

### Evaluate Results

```bash
cd scripts/baselines/generation_scripts/full_rag
./evaluate_full_rag.sh
```

## Input/Output

**Input**: `human/generation_tasks/RAG.jsonl`
**Output**: `results/llama_3.1_8b_full_rag.jsonl`
**Evaluated**: `results/llama_3.1_8b_full_rag_evaluated.jsonl`

## Expected Performance (from Paper Table 5, Llama 3.1 8B)

| Metric | Expected Score |
|--------|---------------|
| RLF (Faithfulness) | 0.56 |
| RBllm (LLM Judge) | 0.59 |
| RBalg (Algorithmic) | 0.34 |
| Answerability Accuracy | 0.74 |

Lower than Reference/Reference+RAG due to retrieval errors.

## Challenges

This setting tests the system's ability to handle:
- **Imperfect retrieval**: Relevant passages may not be in top-5
- **Noisy passages**: Irrelevant information in retrieved results
- **Missing information**: Unanswerable questions where retrieval fails
- **Multi-turn context**: Non-standalone questions require context understanding

## Comparison

- **vs Reference**: Significantly lower (shows retrieval impact)
- **vs Reference+RAG**: Slightly lower (shows impact of missing relevant passages)

This is the **most realistic** and **most challenging** setting, representing actual deployment scenarios.

