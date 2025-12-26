# Two-Stage Retrieval Experiments

This experiment evaluates two-stage retrieval approaches that combine BM25 lexical search with neural reranking.

## Variants

| Script | Stage 1 | Stage 2 | Notes |
|--------|---------|---------|-------|
| `run_two_stage_retrieval.py` | BM25 (local) | BGE (local) | Fully local, no network |
| `run_bm25_elser_rerank.py` | BM25 (local) | ELSER (ES cloud) | Requires ES credentials |

## Hypothesis

A two-stage retrieval pipeline can achieve better retrieval quality than single-stage approaches:
1. **Stage 1 (Recall)**: BM25 retrieves a larger candidate set (top 50) with high recall
2. **Stage 2 (Precision)**: BGE dense model reranks candidates for better precision in the final top 10

This approach balances:
- **Efficiency**: BM25 is fast and doesn't require GPU
- **Quality**: BGE provides semantic matching that lexical methods miss
- **Scalability**: Reranking only 50 documents is much faster than dense retrieval over entire corpus

## Method

### Query Types
We evaluate three query formulations:
- **lastturn**: Only the most recent user question
- **questions**: Concatenation of all questions in conversation
- **rewrite**: Standalone rewritten query (LLM or manual)

### Stage 1: BM25 Filtering
- Uses PyTerrier's BM25 implementation
- Retrieves top 50 candidates per query
- Fast lexical matching for initial filtering

### Stage 2: BGE Reranking  
- Uses `BAAI/bge-base-en-v1.5` sentence transformer
- Encodes query and all 50 candidates
- Computes cosine similarity scores
- Returns top 10 by semantic similarity

### Comparison Baselines
We compare against single-stage baselines:
- **BM25 only**: Top 10 from BM25
- **BGE only**: Top 10 from full corpus BGE retrieval  
- **ELSER**: Elasticsearch's sparse neural retrieval

## Usage

### BM25 + BGE (Local)

```bash
# Run all experiments
python run_two_stage_retrieval.py

# Run specific domain/query type
python run_two_stage_retrieval.py --domain clapnq --query_type rewrite

# Skip existing results
python run_two_stage_retrieval.py --skip-existing
```

### BM25 + ELSER (Requires ES Cloud)

```bash
# Ensure ES credentials are set in .env file:
# ES_URL=https://your-es-cloud-url
# ES_API_KEY=your-api-key

# Run all experiments
python run_bm25_elser_rerank.py

# Run specific domain/query type
python run_bm25_elser_rerank.py --domain clapnq --query_type rewrite

# Adjust rate limiting delay (default 0.3s)
python run_bm25_elser_rerank.py --delay 0.5

# Skip existing results
python run_bm25_elser_rerank.py --skip-existing
```

### Compare with baselines:
```bash
python compare_with_baselines.py
```

## Output Structure

```
intermediate/
├── bm25_top50_{domain}_{query_type}.jsonl      # Stage 1 results
└── bm25_bge_rerank_{domain}_{query_type}.jsonl # Stage 2 results

results/
├── bm25_bge_rerank_{domain}_{query_type}_evaluated.jsonl
├── bm25_bge_rerank_{domain}_{query_type}_evaluated_aggregate.csv
└── comparison_summary.csv                       # Comparison with baselines
```

## Expected Improvements

Based on similar two-stage approaches in the literature:
- **Recall preservation**: BM25@50 should capture most relevant documents
- **Precision boost**: BGE reranking should improve nDCG@10 by filtering noise
- **Complementary signals**: Lexical and semantic signals together are stronger

## Configuration

Key parameters (adjustable in `run_two_stage_retrieval.py`):
- `BM25_TOP_K = 50`: Number of candidates from Stage 1
- `FINAL_TOP_K = 10`: Number of results after reranking
- `BGE_BATCH_SIZE = 32`: Batch size for BGE encoding

