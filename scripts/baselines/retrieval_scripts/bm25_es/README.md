# BM25 Retrieval using Elasticsearch

This directory contains the BM25 retrieval implementation using Elasticsearch for the MT-RAG benchmark.

## Overview

This implementation uses the Elasticsearch Python client to perform BM25 retrieval. Elasticsearch's default scoring algorithm is BM25, making it well-suited for lexical retrieval tasks.

## Requirements

Install the required dependencies:

```bash
pip install elasticsearch
```

You also need to have Elasticsearch running. You can:

### Option 1: Install with Homebrew (macOS - Recommended)

**Install Elasticsearch:**
```bash
brew tap elastic/tap
brew install elastic/tap/elasticsearch-full
```

**Start the Elasticsearch service:**
```bash
brew services start elastic/tap/elasticsearch-full
```

**Verify it's running:**
```bash
curl http://localhost:9200
```

**Stop the service when done:**
```bash
brew services stop elastic/tap/elasticsearch-full
```

**Note for ARM Macs:** If you encounter Java compatibility issues, you may need to install Java 17 and disable ML:
```bash
brew install openjdk@17
# Add to /opt/homebrew/etc/elasticsearch/elasticsearch.yml:
# xpack.ml.enabled: false
```

### Option 2: Run Elasticsearch locally using Docker

```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

### Option 3: Install Elasticsearch directly

Follow the instructions at https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html

## Usage

Run BM25 retrieval on a specific domain:

```bash
python bm25_retrieval.py \
  --domain clapnq \
  --query_type lastturn \
  --corpus_file ../../../corpora/passage_level/clapnq.jsonl \
  --query_file ../../../human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl \
  --output_file results/clapnq_lastturn_results.jsonl \
  --top_k 10
```

### Parameters

- `--domain`: Domain to run retrieval on (choices: clapnq, fiqa, govt, cloud)
- `--query_type`: Type of queries to use (choices: lastturn, rewrite, questions)
- `--corpus_file`: Path to corpus JSONL file
- `--query_file`: Path to queries JSONL file
- `--output_file`: Path to output results file
- `--top_k`: Number of top results to retrieve (default: 10)
- `--es_host`: Elasticsearch host (default: localhost)
- `--es_port`: Elasticsearch port (default: 9200)
- `--index_name`: Elasticsearch index name (default: mtrag_{domain})

## Running All Domains

You can use the provided script to run BM25 retrieval across all domains:

```bash
./run_bm25_es_all.sh
```

## Evaluation

After running retrieval, evaluate the results against ground truth relevance judgments:

```bash
./evaluate_bm25_es.sh
```

This will:
- Evaluate each result file using the qrels (ground truth)
- Generate `*_evaluated.jsonl` files with per-query scores
- Generate `*_evaluated_aggregate.csv` files with aggregated metrics
- Calculate nDCG@1,3,5,10 and Recall@1,3,5,10 for each domain

## Analysis

To analyze and compare results with paper baselines:

```bash
python analyze_results.py
```

This will display:
- Per-domain results for each query type
- Weighted averages across all domains
- Comparison with paper baseline metrics
- Summary table showing differences from baseline

## Implementation Details

- **Indexing**: Documents are indexed into Elasticsearch with BM25 similarity configured
- **Bulk Indexing**: Uses Elasticsearch's bulk API for efficient indexing
- **Query Processing**: Cleans query text by removing special prefixes
- **Retrieval**: Uses Elasticsearch's `match` query on the `text` field
- **Results**: Returns top-k documents with their BM25 scores

## Differences from PyTerrier BM25

- **Backend**: Uses Elasticsearch instead of Terrier
- **Default Parameters**: Elasticsearch BM25 uses k1=1.2 and b=0.75 by default
- **Performance**: Elasticsearch is optimized for production deployments and can handle larger corpora
- **Scalability**: Elasticsearch supports distributed indexing and searching

## Troubleshooting

**Connection Error:**
If you get a connection error, make sure Elasticsearch is running:

For Homebrew installation:
```bash
# Check if service is running
brew services list | grep elasticsearch

# If not running, start it
brew services start elastic/tap/elasticsearch-full

# Verify it's responding
curl http://localhost:9200
```

For Docker:
```bash
# Check if container is running
docker ps | grep elasticsearch

# If not running, start it
docker start elasticsearch

# Verify it's responding
curl http://localhost:9200
```

**Memory Issues:**
For large corpora, you may need to increase Elasticsearch's heap size:
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  elasticsearch:8.11.0
```

