# Targeted Query Rewrite with Mixtral 8x7B

## Purpose

This experiment **isolates the effect of context filtering** by using the same LLM as the paper baseline (Mixtral 8x7B).

The original `targeted_rewrite` experiment used Claude Sonnet 4.5, which introduced a confound: we couldn't tell if performance gains came from:
1. **Better context filtering** (the hypothesis)
2. **Better LLM** (Claude >> Mixtral)

This experiment removes the LLM confound by using Mixtral 8x7B for both:
- Baseline rewrite (full history) — existing paper results
- Targeted rewrite (filtered history) — this experiment

## Hypothesis

If context filtering is genuinely beneficial, we should see improvement even with the same LLM.

## Experimental Design

| Experiment | LLM | Context | Purpose |
|------------|-----|---------|---------|
| **Baseline** (paper) | Mixtral 8x7B | Full history | Control |
| **Targeted Mixtral** (this) | Mixtral 8x7B | Filtered history | Isolate filtering effect |
| **Targeted Claude** (original) | Claude Sonnet 4.5 | Filtered history | Compare LLM effect |

## Key Comparisons

1. **Targeted Mixtral vs Baseline** = Pure context filtering effect
   - Same LLM, different context selection
   - Positive Δ = Filtering genuinely helps

2. **Targeted Claude vs Targeted Mixtral** = LLM quality effect
   - Same filtering, different LLM
   - Shows how much gain came from Claude's superior rewriting

## Method

Same as `targeted_rewrite`:

1. Embed query + history turns with `all-MiniLM-L6-v2`
2. Compute cosine similarity between query and each turn
3. Filter: keep turns with similarity ≥ 0.3, always include last turn, cap at 5
4. Rewrite query with **Mixtral 8x7B** (via Together AI)
5. Run retrieval with BM25, BGE, ELSER
6. Compare against baseline

## Configuration

| Parameter | Value |
|-----------|-------|
| LLM | Mixtral 8x7B Instruct (via Together AI) |
| Similarity threshold | 0.3 |
| Max relevant turns | 5 |
| Include last turn | Always |
| Embedding model | all-MiniLM-L6-v2 |

## Usage

### Prerequisites

Set up Together AI API key in `.env`:
```bash
TOGETHER_API_KEY=your-api-key-here
```

### Step 1: Generate Rewritten Queries

```bash
python run_targeted_rewrite.py --domains clapnq cloud fiqa govt
```

### Step 2: Run Retrieval

```bash
python run_retrieval.py --domains clapnq cloud fiqa govt --retrievers bm25 bge elser
```

### Step 3: Evaluate and Compare

```bash
python evaluate_and_compare.py --run_eval --retriever elser
```

## Expected Results

If context filtering is effective:
- **Targeted Mixtral > Baseline** on most domains
- Gains should be smaller than Targeted Claude (since LLM quality matters too)

If gains were purely from LLM quality:
- **Targeted Mixtral ≈ Baseline** (no improvement)
- All gains would appear in "LLM Effect" column

## Files

```
targeted_rewrite_with_mixtral/
├── run_targeted_rewrite.py      # Generate rewritten queries with Mixtral
├── run_retrieval.py             # Run BM25/BGE/ELSER retrieval
├── evaluate_and_compare.py      # Evaluate and compare results
├── README.md                    # This file
├── intermediate/                # Rewritten queries + analysis
│   ├── targeted_rewrite_mixtral_{domain}.jsonl
│   └── targeted_rewrite_mixtral_{domain}_analysis.json
└── retrieval_results/           # Retrieval + evaluation outputs
    ├── targeted_rewrite_mixtral_{domain}_{retriever}.jsonl
    └── targeted_rewrite_mixtral_{domain}_{retriever}_evaluated_aggregate.csv
```

## API Costs

Mixtral 8x7B via Together AI:
- ~$0.0002/1K tokens (input)
- ~$0.0002/1K tokens (output)
- ~842 queries × ~500 tokens avg = ~$0.08 total (very cheap)

