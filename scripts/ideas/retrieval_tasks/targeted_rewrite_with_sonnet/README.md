# Targeted Query Rewrite with Claude Sonnet

## Purpose

This experiment tests the effect of using a **stronger LLM (Claude Sonnet)** for targeted query rewriting in conversational search. It complements the Mixtral experiment by isolating the effect of LLM quality on the context filtering strategy.

## Hypothesis

Based on the Mixtral experiment findings:
- **Simple similarity-based filtering hurts weaker LLMs** (Mixtral: -3.1% nDCG@10)
- **Stronger LLMs may better infer from less context**, making filtering more effective

This experiment tests whether Claude Sonnet can leverage the targeted (filtered) context better than Mixtral.

## Experimental Design

| Experiment | LLM | Context Strategy |
|------------|-----|------------------|
| **Baseline** (paper) | Mixtral 8x7B | Full conversation history |
| **Targeted Mixtral** | Mixtral 8x7B | Filtered history (similarity ≥ 0.3) |
| **Targeted Sonnet** (this) | Claude Sonnet | Filtered history (similarity ≥ 0.3) |

### Parameters

- **Embedding Model**: `all-MiniLM-L6-v2`
- **LLM Model**: `claude-sonnet-4-20250514`
- **Similarity Threshold**: 0.3
- **Include Last Turn**: Always (regardless of similarity)
- **Top-k Retrieval**: 10 documents

## Full Pipeline Mode

This experiment simulates a **real conversational RAG system**:

```
For each turn:
  1. Filter → Select relevant history turns (semantic similarity)
  2. Rewrite → Rewrite query using filtered context (Claude Sonnet)
  3. Retrieve → Retrieve documents with ELSER (top-10)
  4. Generate → Generate response from retrieved docs (Claude Sonnet)
  5. Update context with GENERATED response (not ground truth)
```

**Why this matters**: Using generated responses (rather than ground truth) for subsequent turns is realistic. Errors in generation can propagate and affect later turns, just as they would in a real system.

## Files

```
targeted_rewrite_with_sonnet/
├── run_full_pipeline.py     # Main pipeline script
├── evaluate_and_compare.py  # Evaluation and comparison
├── utils.py                 # Shared utilities (Anthropic API, retrieval, etc.)
├── README.md                # This file
└── retrieval_results/       # Output directory
    ├── targeted_rewrite_{domain}_elser.jsonl           # Retrieval results
    └── targeted_rewrite_{domain}_elser_analysis.json   # Rewrite analysis
```

## Usage

### Prerequisites

1. **Anthropic API Key**: Set `ANTHROPIC_API_KEY` in `.env`
2. **Elasticsearch**: Set `ES_URL` and `ES_API_KEY` in `.env`
3. **Dependencies**: `pip install sentence-transformers elasticsearch python-dotenv requests anthropic`

### Run the Full Pipeline

```bash
# Run all domains
python run_full_pipeline.py

# Run specific domain
python run_full_pipeline.py --domains clapnq

# Restart from scratch (ignore previous results)
python run_full_pipeline.py --domains clapnq --restart

# Custom similarity threshold
python run_full_pipeline.py --similarity_threshold 0.4
```

The script saves incrementally after each conversation, so you can resume if interrupted.

### Evaluate Results

```bash
# Evaluate and compare
python evaluate_and_compare.py --run_eval --retriever elser
```

This will compare:
- **Targeted Sonnet** (this experiment)
- **Targeted Mixtral** (from sibling experiment)
- **Baseline** (paper's Mixtral with full history)

## Expected Insights

1. **Δ vs Baseline** = Total improvement from using Sonnet + context filtering
   - If positive: Sonnet + filtering outperforms the paper's approach
   - If negative: Even Sonnet can't overcome the filtering's limitations

2. **Δ vs Mixtral Targeted** = Pure effect of LLM quality (same filtering)
   - Shows how much the LLM quality matters for query rewriting
   - Expected to be positive (Sonnet > Mixtral)

3. **% Turns Filtered** = How much context is removed
   - Same filtering strategy as Mixtral (~50% of turns filtered)
   - But Sonnet may handle the reduced context better

## Comparison with Mixtral Experiment

| Metric | Mixtral Targeted | Sonnet Targeted |
|--------|------------------|-----------------|
| LLM Quality | Smaller (8x7B MoE) | Larger (Sonnet) |
| Expected with filtering | Worse (needs more context) | Better (can infer from less) |
| API | Together AI | Anthropic |
| Cost | Lower | Higher |

## Related Work

The literature supports that:
- **LLM quality interacts with context strategy** (your novel finding)
- **Targeted query rewriting is valid** (ConvGQR, CHIQ, LLM-Aided Rewriting)
- **Stronger LLMs can better handle ambiguity** (various QA papers)

