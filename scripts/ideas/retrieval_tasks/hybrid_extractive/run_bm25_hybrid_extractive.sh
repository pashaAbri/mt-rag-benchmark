#!/bin/bash
# Run BM25 retrieval on hybrid extractive rewrite queries

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "======================================"
echo "Running BM25 on Hybrid Extractive Queries"
echo "======================================"
echo ""

# Create results directory
mkdir -p scripts/ideas/retrieval_tasks/hybrid_extractive/results

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running BM25 on $domain with hybrid extractive queries..."
    
    python scripts/baselines/retrieval_scripts/bm25/bm25_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --corpus_file "corpora/passage_level/${domain}.jsonl" \
        --query_file "scripts/ideas/retrieval_tasks/hybrid_extractive/datasets/${domain}_hybrid_extractive.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/hybrid_extractive/results/bm25_${domain}_hybrid_extractive.jsonl" \
        --top_k 10
    
    echo "✓ Completed: bm25_${domain}_hybrid_extractive.jsonl"
    echo ""
done

echo "======================================"
echo "BM25 Retrieval Complete!"
echo "======================================"
echo ""
echo "Results saved to: scripts/ideas/retrieval_tasks/hybrid_extractive/results/"
ls -lh scripts/ideas/retrieval_tasks/hybrid_extractive/results/bm25_*.jsonl
echo ""

