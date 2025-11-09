# Elser Sparse Retrieval

Elser (ElasticSearch Learned Sparse EncodeR) retrieval implementation for the MT-RAG benchmark.

## Model

- **Name:** ELSERv1
- **Type:** Learned sparse retrieval
- **Platform:** ElasticSearch 8.10+
- **Documentation:** https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html

## Status

✅ **COMPLETE** for all 4 domains (ClapNQ, FiQA, Cloud, Govt)
✅ **All 777 queries** across 12 experiments (4 domains × 3 query types)

## Requirements

- ElasticSearch 8.10 or higher
- ELSERv1 model deployed in ElasticSearch
- Python elasticsearch client

## Usage

### Prerequisites

Set up Elasticsearch Cloud credentials in `.env` file:
```bash
ES_URL=https://your-deployment.es.region.cloud.es.io:443
ES_API_KEY=your-api-key
```

### Run Single Experiment

```bash
python scripts/baselines/retrieval_scripts/elser/elser_retrieval.py \
    --domain fiqa \
    --query_type lastturn \
    --query_file human/retrieval_tasks/fiqa/fiqa_lastturn.jsonl \
    --output_file scripts/baselines/retrieval_scripts/elser/results/elser_fiqa_lastturn.jsonl \
    --top_k 10 \
    --delay 2.0
```

**Parameters:**
- `--domain`: Domain (clapnq, fiqa, govt, cloud)
- `--query_type`: Query type (lastturn, rewrite, questions)
- `--delay`: Delay between queries in seconds (default: 0.5, recommended: 2.0 to avoid rate limits)

### Run All Experiments

```bash
bash scripts/baselines/retrieval_scripts/elser/run_elser_all.sh
```

Runs ELSER on all 4 domains × 3 query types = 12 experiments
**Time:** ~52 minutes with 2-second delays (777 queries total)

### Evaluate Results

```bash
bash scripts/baselines/retrieval_scripts/elser/evaluate_elser.sh
```

## Implementation Notes

### Architecture
- **Backend:** Elasticsearch Cloud Serverless (v8.11.0)
- **Model:** ELSERv2 via `.elser-2-elastic` inference endpoint
- **Indexing:** Documents indexed with ELSER pipeline (512 token chunks, 100 token overlap)
- **Query Method:** `text_expansion` query type (deprecated but functional)
- **Field:** Queries against `ml.tokens` sparse vector field

### Rate Limiting
- Elasticsearch Cloud enforces strict rate limits on ELSER inference
- **Recommended delay:** 2 seconds between queries
- Built-in retry logic with exponential backoff
- Handles 429 rate limit errors gracefully

### Performance
- **Query speed:** ~2-3 seconds per query (including delay)
- **Throughput:** ~30 queries/minute
- **Total time for all 4 domains:** ~52 minutes (2,331 queries total)

## Results

See `results/summary_elser.md` for detailed performance analysis.

### Quick Summary (All 4 domains: ClapNQ, FiQA, Cloud, Govt)

**Last Turn (Paper's reported metric):**
- Recall@5: 0.439
- nDCG@5: 0.405

**Query Rewrite (Best performing):**
- Recall@5: 0.476
- nDCG@5: 0.438

**Comparison with Paper Baseline (Table 1):**
- Paper's ELSER Last Turn: R@5=0.49, nDCG@5=0.45
- Our ELSER Last Turn: R@5=0.44, nDCG@5=0.41
- **Difference: -0.05 R@5, -0.04 nDCG@5** ✅ Very close match!

### Domain Performance (Recall@5 with Query Rewrite)
1. **ClapNQ:** 0.552 (best) 🏆
2. **Govt:** 0.508
3. **Cloud:** 0.430
4. **FiQA:** 0.402 (hardest domain)

## Troubleshooting

### Rate Limit Errors

If you see `429 Rate limit exceeded` errors:
- Increase `--delay` parameter (try 3.0 or 5.0 seconds)
- Use the retry script for failed queries:
```bash
python scripts/baselines/retrieval_scripts/elser/retry_failed_queries.py \
    --domain fiqa \
    --query_type questions \
    --query_file human/retrieval_tasks/fiqa/fiqa_questions.jsonl \
    --failed_ids_file fiqa_questions_failed_ids.txt \
    --output_file scripts/baselines/retrieval_scripts/elser/results/elser_fiqa_questions.jsonl \
    --delay 5.0
```

### ClapNQ Index Name

**Note:** ClapNQ uses the reindexed version:
- Index name: `mtrag-clapnq-elser-512-100-reindexed`
- This is automatically handled by the `elser_retrieval.py` script
- All 183,408 documents fully indexed with ELSER tokens ✅

## Files

- `elser_retrieval.py` - Main ELSER retrieval script
- `retry_failed_queries.py` - Retry failed queries due to rate limits
- `reindex_clapnq.py` - Re-index ClapNQ with ELSER pipeline
- `run_elser_all.sh` - Run all experiments (3 domains)
- `evaluate_elser.sh` - Evaluate all results
- `results/` - Retrieval and evaluation outputs
- `results/summary_elser.md` - Detailed performance analysis

