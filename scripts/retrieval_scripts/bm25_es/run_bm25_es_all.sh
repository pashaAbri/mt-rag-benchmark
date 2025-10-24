#!/bin/bash

# Run BM25 Elasticsearch retrieval for all domains and query types

# Create results directory if it doesn't exist
mkdir -p results

# Define domains and query types
DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

# Elasticsearch configuration
ES_HOST="localhost"
ES_PORT=9200

# Run retrieval for each combination
for domain in "${DOMAINS[@]}"; do
  for query_type in "${QUERY_TYPES[@]}"; do
    echo "========================================"
    echo "Running BM25 ES retrieval for $domain with $query_type queries..."
    echo "========================================"
    
    python bm25_retrieval.py \
      --domain "$domain" \
      --query_type "$query_type" \
      --corpus_file "../../../corpora/passage_level/${domain}.jsonl" \
      --query_file "../../../human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
      --output_file "results/${domain}_${query_type}_results.jsonl" \
      --top_k 10 \
      --es_host "$ES_HOST" \
      --es_port "$ES_PORT"
    
    if [ $? -ne 0 ]; then
      echo "Error running retrieval for $domain with $query_type queries"
      exit 1
    fi
  done
done

echo "========================================"
echo "All BM25 ES retrieval runs completed!"
echo "========================================"

