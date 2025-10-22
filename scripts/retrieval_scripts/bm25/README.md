# BM25 Retrieval

BM25 lexical retrieval implementation using PyTerrier for the MT-RAG benchmark.

## Files

- `bm25_retrieval.py` - Main BM25 retrieval script
- `run_bm25_all.sh` - Run BM25 on all 4 domains × 3 query types (12 experiments)
- `evaluate_bm25.sh` - Evaluate all BM25 results against ground truth
- `results/` - Directory containing all BM25 results and evaluations

## Usage

### Run single experiment:

```bash
python scripts/retrieval_scripts/bm25/bm25_retrieval.py \
    --domain clapnq \
    --query_type lastturn \
    --corpus_file corpora/passage_level/clapnq.jsonl \
    --query_file human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl \
    --output_file scripts/retrieval_scripts/bm25/results/bm25_clapnq_lastturn.jsonl \
    --top_k 10
```

### Run all experiments:

```bash
bash scripts/retrieval_scripts/bm25/run_bm25_all.sh
```

### Evaluate all results:

```bash
bash scripts/retrieval_scripts/bm25/evaluate_bm25.sh
```

## Results

See `results/summary_bm25.md` for detailed comparison with paper baselines.

Key findings:
- Query Rewrite outperforms Last Turn (R@5: 0.261 vs 0.234)
- Our results match/exceed paper baselines
- Govt domain performs best, FiQA performs worst

