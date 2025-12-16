# Mono-T5 Reranker-Targeted Full RAG Generation

This experiment evaluates generation quality when using the **best retrieval results** from the mono-t5 reranker-targeted pipeline (3-strategy: LastTurn + Questions + Targeted Rewrite).

## Overview

- **Retrieval Method**: Mono-T5 reranker with targeted rewrite (3-strategy combination)
- **Retrieval Performance**: Best in benchmark (nDCG@10: 0.5333)
- **Tasks**: 842 (all tasks from benchmark)
- **Passages**: Top 5 from mono-t5 reranked results
- **Purpose**: Test if better retrieval leads to better generation

## Retrieval Strategy

The retrieval pipeline combines three query strategies:

1. **LastTurn**: The last user turn in the conversation
2. **Questions**: All questions from the conversation concatenated
3. **Targeted Rewrite**: Query rewrite using filtered conversation history (semantic similarity filtering)

Documents are retrieved using ELSER, then **reranked with MonoT5** using the targeted rewrite query.

### Retrieval Performance (for reference)

| Metric  | Score      |
| ------- | ---------- |
| R@1     | 0.2083     |
| R@3     | 0.4142     |
| R@5     | 0.5112     |
| R@10    | 0.6434     |
| nDCG@10 | **0.5333** |

## Structure

```
mono-t5-as-reranker-targeted-full-rag/
├── README.md                      # This file
├── prepare_generation_tasks.py    # Creates generation tasks from retrieval results
├── run_generation.sh              # Runs generation with GPT-4o
├── evaluate_generation.sh         # Evaluates generation results
├── mono_t5_targeted_RAG.jsonl     # Generated tasks file (created by prepare script)
└── results/
    ├── gpt_4o_mono_t5_targeted_full_rag.jsonl           # Generation results
    └── gpt_4o_mono_t5_targeted_full_rag_evaluated.jsonl # Evaluated results
```

## Usage

### Step 1: Prepare Generation Tasks

```bash
# From project root
python scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag/prepare_generation_tasks.py
```

This creates `mono_t5_targeted_RAG.jsonl` by:

1. Loading existing RAG.jsonl as template (conversation history, targets, enrichments)
2. Loading mono-t5 retrieval results for all 4 domains
3. Loading corpus files to get document text
4. Replacing contexts with top-5 reranked documents

### Step 2: Run Generation

```bash
cd scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag
./run_generation.sh
```

Or run both steps together (the script checks if preparation is needed):

```bash
./run_generation.sh
```

### Step 3: Evaluate Results

```bash
./evaluate_generation.sh
```

## Environment Variables

Required in `.env` file:

- `OPENAI_API_KEY`: For GPT-4o generation and evaluation

## Comparison with Baseline

The baseline full_rag results use:

- **Retrieval**: ELSER with query rewrite (no reranking)
- **Performance**: nDCG@10 ≈ 0.495

Our approach improves retrieval by:

- Adding mono-t5 reranking (+7.7% nDCG@10)
- Using targeted rewrite queries (filtered conversation history)

## Expected Improvements

Since retrieval quality directly impacts generation:

1. **Higher faithfulness**: Better passages = more grounded responses
2. **Better answerability detection**: Relevant documents help identify unanswerable questions
3. **Improved reference-based scores**: More relevant context = better match to reference answers

## Data Sources

- **Retrieval results**: `scripts/ideas/retrieval_tasks/mono-t5-as-reranker-targeted/intermediate/using_targeted_rewrite_query/`
- **Corpus files**: `corpora/passage_level/` (clapnq, cloud, fiqa, govt)
- **Template tasks**: `human/generation_tasks/RAG.jsonl`

## Models Tested

- [x] GPT-4o (gpt-4o)
- [ ] Llama 3.1 8B
- [ ] Llama 3.1 70B
- [ ] Llama 3.1 405B
- [ ] Others (can be added)

## Results

_Results will be added here after running experiments_

| Model  | Ans. Acc. | RLF | RBllm | RBalg |
| ------ | --------- | --- | ----- | ----- |
| GPT-4o | TBD       | TBD | TBD   | TBD   |

### Comparison with Baseline Full RAG

| Setting                     | Model  | Ans. Acc. | RLF | RBllm | RBalg |
| --------------------------- | ------ | --------- | --- | ----- | ----- |
| Baseline Full RAG (ELSER)   | GPT-4o | TBD       | TBD | TBD   | TBD   |
| **Mono-T5 Targeted (Ours)** | GPT-4o | TBD       | TBD | TBD   | TBD   |
