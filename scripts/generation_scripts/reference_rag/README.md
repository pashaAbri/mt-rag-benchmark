# Reference+RAG Setting - Hybrid Retrieval

This scenario tests generation with **gold passages mixed with retrieved passages**, simulating the upper bound for real RAG systems.

## Overview

- **Setting**: Reference+RAG (⊕)
- **Tasks**: 436 (subset with ≤2 reference passages)
- **Passages**: Gold reference + retrieved passages (total 5)
- **Purpose**: Test noise handling with guaranteed correct passages

## Retrieval Strategy

- **Gold reference passages** are included (up to 2)
- **Retrieved passages** fill remaining slots to reach 5 total
- Uses **Elser with query rewrite** for retrieval
- Guarantees all necessary information is present, but adds noise

**Why subset?** Only tasks with ≤2 reference passages are included so all gold passages fit in top-5.

## Usage

### Run Generation

```bash
cd scripts/generation_scripts/reference_rag
./run_reference_rag.sh
```

### Evaluate Results

```bash
cd scripts/generation_scripts/reference_rag
./evaluate_reference_rag.sh
```

## Input/Output

**Input**: `human/generation_tasks/reference+RAG.jsonl`
**Output**: `results/llama_3.1_8b_reference_rag.jsonl`
**Evaluated**: `results/llama_3.1_8b_reference_rag_evaluated.jsonl`

## Expected Performance (from Paper Table 5, Llama 3.1 8B)

| Metric | Expected Score |
|--------|---------------|
| RLF (Faithfulness) | 0.56 |
| RBllm (LLM Judge) | 0.59 |
| RBalg (Algorithmic) | 0.35 |
| Answerability Accuracy | 0.75 |

Slightly lower than Reference due to added noise from retrieved passages.

## Comparison

- **vs Reference**: Minimal degradation (noise effect is small)
- **vs Full RAG**: Better performance (guarantees relevant passages are present)

This setting represents the **best case for real RAG** - retrieval succeeds but includes some irrelevant passages.

