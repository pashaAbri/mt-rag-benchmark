# Targeted Query Rewrite with Mixtral 8x7B

## Purpose

This experiment **isolates the effect of context filtering** by using the same LLM as the paper baseline (Mixtral 8x7B).

The original `targeted_rewrite` experiment used Claude Sonnet 4.5, which introduced a confound: we couldn't tell if performance gains came from:

1. **Better context filtering** (the hypothesis)
2. **Better LLM** (Claude >> Mixtral)

This experiment removes the LLM confound by using Mixtral 8x7B for both:

- Baseline rewrite (full history) — existing paper results
- Targeted rewrite (filtered history) — this experiment

## Full Pipeline

The `run_full_pipeline.py` script simulates a **real conversational RAG system**:

```
For each turn:
    1. Filter history (semantic similarity) ← TARGETED REWRITE
    2. Rewrite query (Mixtral)
    3. Retrieve documents (ELSER)           ← RETRIEVAL
    4. Generate response (Mixtral)          ← GENERATION
    5. Add GENERATED response to context    ← Not ground truth!
```

This is the correct approach because in a real system:
- The agent's responses come from retrieval + generation (not pre-existing ground truth)
- Each turn's context includes the actual system's behavior, not ideal responses
- This properly tests whether targeted rewrite helps in realistic conditions

## Configuration

| Parameter            | Value                                   |
| -------------------- | --------------------------------------- |
| LLM                  | Mixtral 8x7B Instruct (via Together AI) |
| Retriever            | ELSER v2 (Elasticsearch)                |
| Similarity threshold | 0.3                                     |
| Max relevant turns   | Unlimited (all above threshold)         |
| Include last turn    | Always                                  |
| Embedding model      | all-MiniLM-L6-v2                        |
| Processing           | Sequential (turn-by-turn)               |

## Usage

### Prerequisites

Set up API keys in `.env` at project root:

```bash
TOGETHER_API_KEY=your-together-api-key
ES_URL=your-elasticsearch-url
ES_API_KEY=your-elasticsearch-api-key
```

### Run the Experiment

```bash
cd scripts/ideas/retrieval_tasks/targeted_rewrite_with_mixtral

# Step 1: Run full pipeline (rewrite → retrieve → generate)
python run_full_pipeline.py --domains clapnq cloud fiqa govt

# Step 2: Evaluate and compare against baseline
python evaluate_and_compare.py --run_eval --retriever elser
```

### Options

```bash
python run_full_pipeline.py \
    --domains clapnq cloud fiqa govt \
    --similarity_threshold 0.3 \
    --top_k 10 \
    --skip_existing
```

## Files

```
targeted_rewrite_with_mixtral/
├── run_full_pipeline.py         # Full pipeline (rewrite → retrieve → generate)
├── evaluate_and_compare.py      # Evaluate and compare against baselines
├── utils.py                     # Shared utilities (LLM, retrieval, data loading)
├── README.md                    # This file
└── retrieval_results/           # Output: retrieval results + analysis
    ├── targeted_rewrite_{domain}_elser.jsonl
    └── targeted_rewrite_{domain}_elser_analysis.json
```

## Output Format

### Retrieval Results (`*_elser.jsonl`)

```json
{
    "task_id": "conv_id<::>turn_id",
    "Collection": "mt-rag-clapnq-elser-512-100-20240503",
    "contexts": [
        {"document_id": "...", "score": 12.34, "text": "...", "title": "..."},
        ...
    ]
}
```

### Analysis (`*_analysis.json`)

```json
{
    "config": {"similarity_threshold": 0.3, "processing_mode": "full_pipeline", ...},
    "stats": {"turn1_no_rewrite": 30, "rewritten": 180, "turns_filtered": 120},
    "analyses": [
        {
            "task_id": "...",
            "turn_id": 3,
            "original_query": "What about its diet?",
            "rewritten_query": "What is the diet of Cavia tschudii in the wild?",
            "generated_response": "Based on the documents...",
            "method": "targeted_rewrite",
            "num_history_turns": 2,
            "selected_turns": 1
        }
    ]
}
```

## API Costs

Mixtral 8x7B via Together AI:

- ~$0.0002/1K tokens (input)
- ~$0.0002/1K tokens (output)
- ~842 queries × 2 calls (rewrite + generate) × ~500 tokens = ~$0.17 total

## Key Insight

The comparison tells us:

| Comparison | Meaning |
|------------|---------|
| Δ nDCG > 0 | Context filtering **helps** — less noise in rewritten queries |
| Δ nDCG < 0 | Context filtering **hurts** — over-filtering useful context |
| Δ nDCG ≈ 0 | Context filtering has **no effect** |

Since both use the same LLM (Mixtral), any difference is purely from context filtering.
