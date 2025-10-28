# Elser Sparse Retrieval

Elser (ElasticSearch Learned Sparse EncodeR) retrieval implementation for the MT-RAG benchmark.

## Model

- **Name:** ELSERv1
- **Type:** Learned sparse retrieval
- **Platform:** ElasticSearch 8.10+
- **Documentation:** https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html

## Status

✅ **COMPLETE** for 3/4 domains (FiQA, Cloud, Govt)
⏳ **ClapNQ:** Reindexing in progress (~12-15 hours)

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
python scripts/retrieval_scripts/elser/elser_retrieval.py \
    --domain fiqa \
    --query_type lastturn \
    --query_file human/retrieval_tasks/fiqa/fiqa_lastturn.jsonl \
    --output_file scripts/retrieval_scripts/elser/results/elser_fiqa_lastturn.jsonl \
    --top_k 10 \
    --delay 2.0
```

**Parameters:**
- `--domain`: Domain (fiqa, govt, cloud) - ClapNQ pending reindex
- `--query_type`: Query type (lastturn, rewrite, questions)
- `--delay`: Delay between queries in seconds (default: 0.5, recommended: 2.0 to avoid rate limits)

### Run All Experiments

```bash
bash scripts/retrieval_scripts/elser/run_elser_all.sh
```

Runs ELSER on all 3 working domains × 3 query types = 9 experiments
**Time:** ~45 minutes with 2-second delays

### Evaluate Results

```bash
bash scripts/retrieval_scripts/elser/evaluate_elser.sh
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
- **Total time for 3 domains:** ~45 minutes (1,707 queries)

## Results

See `results/summary_elser.md` for detailed performance analysis.

### Quick Summary (3 domains: FiQA, Cloud, Govt)

**Query Rewrite (Best performing):**
- Recall@5: 0.449
- nDCG@5: 0.410

**Comparison with Paper Baseline:**
- Our 3-domain average: R@5=0.45, nDCG@5=0.41
- Paper's 4-domain average: R@5=0.52, nDCG@5=0.48
- Govt domain alone: R@5=0.51, nDCG@5=0.45 ✅ Matches paper!

### Domain Performance (Recall@5 with Query Rewrite)
1. **Govt:** 0.508 (best)
2. **Cloud:** 0.430
3. **FiQA:** 0.402 (hardest domain)

## Troubleshooting

### Rate Limit Errors

If you see `429 Rate limit exceeded` errors:
- Increase `--delay` parameter (try 3.0 or 5.0 seconds)
- Use the retry script for failed queries:
```bash
python scripts/retrieval_scripts/elser/retry_failed_queries.py \
    --domain fiqa \
    --query_type questions \
    --query_file human/retrieval_tasks/fiqa/fiqa_questions.jsonl \
    --failed_ids_file fiqa_questions_failed_ids.txt \
    --output_file scripts/retrieval_scripts/elser/results/elser_fiqa_questions.jsonl \
    --delay 5.0
```

### ClapNQ Not Ready

ClapNQ is currently being reindexed with ELSER tokens. Check progress:
```bash
python -c "from elasticsearch import Elasticsearch; import os; from dotenv import load_dotenv; load_dotenv(); es = Elasticsearch(os.getenv('ES_URL'), api_key=os.getenv('ES_API_KEY')); print(f'Progress: {es.count(index=\"mtrag-clapnq-elser-512-100-reindexed\")[\"count\"]:,} / 183,408')"
```

## Files

- `elser_retrieval.py` - Main ELSER retrieval script
- `retry_failed_queries.py` - Retry failed queries due to rate limits
- `reindex_clapnq.py` - Re-index ClapNQ with ELSER pipeline
- `run_elser_all.sh` - Run all experiments (3 domains)
- `evaluate_elser.sh` - Evaluate all results
- `results/` - Retrieval and evaluation outputs
- `results/summary_elser.md` - Detailed performance analysis

