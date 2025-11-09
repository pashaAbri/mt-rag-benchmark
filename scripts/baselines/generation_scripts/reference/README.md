# Reference Setting - Perfect Retriever

This scenario simulates a **perfect retriever** by using gold-standard reference passages curated by human annotators.

## Overview

- **Setting**: Reference (•)
- **Tasks**: 842 (all tasks from benchmark)
- **Passages**: Human-curated gold passages (no retrieval)
- **Purpose**: Test generator capability in isolation (upper bound)

## Retrieval Strategy

- For **answerable/partial** questions: Gold reference passages provided
- For **unanswerable** questions: NO passages provided
- For **conversational** questions: NO passages provided

This represents the **ideal case** where retrieval is perfect.

## Usage

### Run Generation

```bash
cd scripts/baselines/generation_scripts/reference
./run_reference.sh
```

### Evaluate Results

```bash
cd scripts/baselines/generation_scripts/reference
./evaluate_reference.sh
```

## Input/Output

**Input**: `human/generation_tasks/reference.jsonl`
**Output**: `results/llama_3.1_8b_reference.jsonl`
**Evaluated**: `results/llama_3.1_8b_reference_evaluated.jsonl`

## Expected Performance (from Paper Table 5, Llama 3.1 8B)

| Metric | Expected Score |
|--------|---------------|
| RLF (Faithfulness) | 0.55 |
| RBllm (LLM Judge) | 0.59 |
| RBalg (Algorithmic) | 0.36 |
| Answerability Accuracy | 0.71 |

This is the **best possible** performance since retrieval is perfect.

## Comparison with Other Settings

Reference provides the **upper bound** on generation quality. Comparing with other settings shows the impact of imperfect retrieval:

- **Reference** → **Reference+RAG**: Small degradation (adding noise)
- **Reference** → **Full RAG**: Larger degradation (real retrieval errors)

