# BGE-base 1.5 Dense Retrieval

BGE (BAAI General Embedding) dense retrieval implementation for the MT-RAG benchmark.

## Model

- **Name:** BAAI/bge-base-en-v1.5
- **Type:** Dense retrieval using neural embeddings
- **Paper:** C-Pack: Packaged resources to advance general chinese embedding (Xiao et al., 2023)
- **Size:** ~420 MB
- **Embedding Dimension:** 768

## Setup

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install sentence-transformers torch
```

### 2. Download Model Locally

We download the model to a local `models/` directory (ignored by git):

```bash
cd scripts/baselines/retrieval_scripts/bge
python download_model.py
```

This will:
- Download BAAI/bge-base-en-v1.5 (~420 MB)
- Save to `scripts/baselines/retrieval_scripts/bge/models/bge-base-en-v1.5/`
- Allow offline usage after first download

**Note:** The `models/` directory is gitignored to avoid committing large model files.

## Usage

### Run Single Experiment

```bash
python scripts/baselines/retrieval_scripts/bge/bge_retrieval.py \
    --domain clapnq \
    --query_type lastturn \
    --corpus_file corpora/passage_level/clapnq.jsonl \
    --query_file human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl \
    --output_file scripts/baselines/retrieval_scripts/bge/results/bge_clapnq_lastturn.jsonl \
    --top_k 10
```

### Parameters

- `--domain`: Domain to run retrieval on (choices: clapnq, fiqa, govt, cloud)
- `--query_type`: Type of queries to use (choices: lastturn, rewrite, questions)
- `--corpus_file`: Path to corpus JSONL file
- `--query_file`: Path to queries JSONL file
- `--output_file`: Path to output results file
- `--top_k`: Number of top results to retrieve (default: 10)
- `--model_path`: Path to local BGE model (default: models/bge-base-en-v1.5)
- `--batch_size`: Batch size for encoding (default: 64)

### Run All Domains

Run BGE retrieval across all domains and query types:

```bash
./run_bge_all.sh
```

This runs 12 experiments (4 domains × 3 query types).

### Evaluation

After running retrieval, evaluate the results:

```bash
./evaluate_bge.sh
```

This will:
- Evaluate each result file using the qrels (ground truth)
- Generate `*_evaluated.jsonl` files with per-query scores
- Generate `*_evaluated_aggregate.csv` files with aggregated metrics
- Calculate nDCG@1,3,5,10 and Recall@1,3,5,10 for each domain

### Analysis

Analyze and compare results with paper baselines:

```bash
python analyze_results.py
```

This will display:
- Per-domain results for each query type
- Weighted averages across all domains
- Comparison with paper baseline metrics (Table 3)
- Summary table showing differences from baseline

## Implementation Details

**Model:**
- Sentence-transformers library
- BAAI/bge-base-en-v1.5 from HuggingFace
- Embedding dimension: 768
- Similarity: Cosine similarity (via normalized inner product)

**Performance Optimization:**
- **Device**: Auto-detects MPS (Mac GPU), CUDA (NVIDIA), or CPU
- **Embeddings Caching**: Documents are encoded once per domain and cached to `embeddings/{domain}_embeddings.npy`
- **FAISS**: Efficient similarity search (IndexFlatIP for inner product)
- **Batch Encoding**: Processes documents and queries in batches

**Caching Strategy:**
- First run: Encodes all documents (~5-15 minutes per domain) and saves to disk
- Subsequent runs: Loads cached embeddings instantly
- Cache location: `scripts/baselines/retrieval_scripts/bge/embeddings/`
- Cache size: ~562 MB per domain (183K docs × 768 dim × 4 bytes)
- Cache is gitignored (not committed to repo)

**Query Processing:**
- Removes `|user|:` prefix from queries (same as BM25)
- Encodes queries on every run (fast, only ~200 queries)

## Expected Performance (from Paper Table 3)

BGE-base 1.5 is expected to significantly outperform BM25:

| Query Strategy | R@5 | R@10 | nDCG@5 | nDCG@10 |
|----------------|-----|------|--------|---------|
| **Last Turn** | 0.30 | 0.38 | 0.27 | 0.30 |
| **Query Rewrite** | 0.37 | 0.47 | 0.34 | 0.38 |

**Comparison with BM25:**
- Last Turn R@5: 0.30 (BGE) vs 0.20 (BM25) → +50%
- Query Rewrite R@5: 0.37 (BGE) vs 0.25 (BM25) → +48%

BGE's dense embeddings capture semantic meaning, leading to better retrieval performance than keyword-based BM25.

## Troubleshooting

**Model Not Found:**
```bash
# Run the download script first
python download_model.py
```

**Out of Memory:**
```bash
# Reduce batch size
python bge_retrieval.py ... --batch_size 16
```

**MPS/GPU Issues:**
If you encounter issues with MPS (Mac GPU), the script will automatically fall back to CPU. You can also force CPU by setting:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Slow Encoding:**
- First run per domain takes 5-15 minutes to encode all documents
- Subsequent runs load cached embeddings instantly
- Check `embeddings/` directory for cached files

