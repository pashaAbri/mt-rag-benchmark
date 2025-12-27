# Baseline Query Rewrite with Claude Sonnet (Full History)

## Purpose

This experiment establishes a **fair baseline** for Claude Sonnet by using **FULL conversation history** (no filtering). This allows us to isolate the effect of context filtering from the effect of LLM quality.

## Problem with Previous Comparison

In the previous experiments, we compared:
- **Baseline**: Mixtral with full history
- **Targeted Sonnet**: Sonnet with filtered history

This confounds TWO variables:
1. LLM quality (Mixtral vs Sonnet)
2. Context strategy (full vs filtered)

## Solution: Fair Baseline

This experiment adds:
- **Sonnet Full**: Sonnet with full history (NEW)

Now we can isolate each variable:

| Comparison | What it measures |
|------------|------------------|
| Sonnet Full vs Mixtral Full | Pure LLM quality effect |
| Sonnet Filtered vs Sonnet Full | Pure filtering effect |

## Experimental Design

| Experiment | LLM | Context Strategy |
|------------|-----|------------------|
| **Mixtral Full** (paper) | Mixtral 8x7B | Full history |
| **Sonnet Full** (this) | Claude Sonnet | Full history |
| **Sonnet Filtered** | Claude Sonnet | Filtered history |

### Parameters

- **LLM Model**: `claude-sonnet-4-20250514`
- **Context Strategy**: FULL conversation history (no filtering)
- **Top-k Retrieval**: 10 documents

## Full Pipeline Mode

This experiment uses the same full pipeline as targeted rewrite:

```
For each turn:
  1. Rewrite → Use FULL conversation history (no filtering)
  2. Retrieve → Retrieve documents with ELSER (top-10)
  3. Generate → Generate response from retrieved docs
  4. Update context with GENERATED response (not ground truth)
```

## Files

```
baseline_rewrite_with_sonnet/
├── run_full_pipeline.py     # Main pipeline script (full history)
├── evaluate_and_compare.py  # Evaluation and comparison
├── utils.py                 # Shared utilities
├── README.md                # This file
└── retrieval_results/       # Output directory
```

## Usage

### Prerequisites

1. **Anthropic API Key**: Set `ANTHROPIC_API_KEY` in `.env`
2. **Elasticsearch**: Set `ES_URL` and `ES_API_KEY` in `.env`

### Run the Pipeline

```bash
# Run all domains
python run_full_pipeline.py

# Run specific domain
python run_full_pipeline.py --domains clapnq

# Restart from scratch
python run_full_pipeline.py --domains clapnq --restart
```

### Evaluate Results

```bash
python evaluate_and_compare.py --run_eval --retriever elser
```

## Expected Insights

1. **Δ Sonnet Full vs Mixtral Full** = Pure LLM quality effect
   - Expected: Sonnet should outperform Mixtral

2. **Δ Sonnet Filtered vs Sonnet Full** = Pure filtering effect
   - If negative: Filtering hurts even with Sonnet
   - If positive: Filtering helps with Sonnet
   - If ~0: Filtering is neutral

This will definitively answer whether context filtering is beneficial when using a strong LLM.

