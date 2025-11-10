#!/bin/bash
# Run ELSER retrieval on pure extractive rewrite queries

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "======================================"
echo "Running ELSER on Pure Extractive Queries"
echo "======================================"
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running ELSER on $domain with pure extractive queries..."
    
    python scripts/baselines/retrieval_scripts/elser/elser_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --query_file "scripts/ideas/retrieval_tasks/pure_extractive/datasets/${domain}_pure_extractive.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/pure_extractive/results/elser_${domain}_pure_extractive.jsonl" \
        --top_k 10 \
        --delay 2.0
    
    echo "✓ Completed: elser_${domain}_pure_extractive.jsonl"
    echo ""
done

echo "======================================"
echo "ELSER Retrieval Complete!"
echo "======================================"
echo ""
echo "Results saved to: scripts/ideas/retrieval_tasks/pure_extractive/results/"
ls -lh scripts/ideas/retrieval_tasks/pure_extractive/results/elser_*.jsonl
echo ""

