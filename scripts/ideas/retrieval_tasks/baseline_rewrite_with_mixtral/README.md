# Baseline Query Rewrite with Mixtral 8x7B (Full History)

## Purpose

This experiment establishes a **baseline for Mixtral 8x7B** using **FULL conversation history** (no filtering) with our full pipeline approach. This replicates the paper's baseline strategy but uses our complete pipeline (rewrite → retrieve → generate) with generated responses.

## Comparison with Paper Baseline

The paper's "Mixtral Full" baseline uses:
- Pre-computed rewritten queries from dataset files
- Does NOT run live rewrite → retrieve → generate pipeline

This experiment:
- Runs live Mixtral 8x7B calls for both rewrite and generate
- Uses generated responses (not ground truth) for context in subsequent turns
- Provides a fair comparison with our Sonnet experiments

## Experimental Design

| Experiment | LLM | Context Strategy | Pipeline |
|------------|-----|------------------|----------|
| **Paper Baseline** | Mixtral 8x7B | Full history | Pre-computed rewrites |
| **Mixtral Full** (this) | Mixtral 8x7B | Full history | Live pipeline |
| **Sonnet Full** | Claude Sonnet | Full history | Live pipeline |
| **Sonnet Filtered** | Claude Sonnet | Filtered history | Live pipeline |

### Parameters

- **LLM Model**: `mistralai/Mixtral-8x7B-Instruct-v0.1` (via Together AI)
- **Context Strategy**: FULL conversation history (no filtering)
- **Top-k Retrieval**: 10 documents

## Full Pipeline Mode

This experiment uses the same full pipeline as other experiments:

```
For each turn:
  1. Rewrite → Use FULL conversation history (no filtering)
  2. Retrieve → Retrieve documents with ELSER (top-10)
  3. Generate → Generate response from retrieved docs
  4. Update context with GENERATED response (not ground truth)
```

## Files

```
baseline_rewrite_with_mixtral/
├── run_full_pipeline.py     # Main pipeline script (full history)
├── evaluate_and_compare.py  # Evaluation and comparison (copy from sonnet)
├── utils.py                 # Shared utilities
├── README.md                # This file
└── retrieval_results/       # Output directory
```

## Usage

### Prerequisites

1. **Together AI API Key**: Set `TOGETHER_API_KEY` in `.env`
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

Copy the evaluation script from `baseline_rewrite_with_sonnet/` and run:

```bash
python evaluate_and_compare.py --run_eval --retriever elser
```

## Expected Insights

This experiment helps answer:

1. **Does our pipeline match the paper baseline?**
   - Compare our Mixtral Full with paper's pre-computed results
   - Any difference shows the impact of live pipeline vs pre-computed

2. **LLM quality effect (Mixtral vs Sonnet)**
   - Compare Mixtral Full vs Sonnet Full (both with full history)
   - Isolates the pure LLM quality effect

3. **Context filtering effect**
   - Compare Sonnet Full vs Sonnet Filtered
   - Shows whether filtering helps or hurts with Sonnet

