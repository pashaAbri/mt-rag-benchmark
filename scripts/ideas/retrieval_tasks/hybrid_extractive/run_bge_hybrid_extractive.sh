#!/bin/bash
# Run BGE dense retrieval on hybrid extractive rewrite queries

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "======================================"
echo "Running BGE on Hybrid Extractive Queries"
echo "======================================"
echo ""

# Use the baseline BGE model that's already downloaded
BGE_MODEL_PATH="scripts/baselines/retrieval_scripts/bge/models/bge-base-en-v1.5"

if [ ! -d "$BGE_MODEL_PATH" ]; then
    echo "ERROR: BGE model not found at $BGE_MODEL_PATH"
    echo "Please run: cd scripts/baselines/retrieval_scripts/bge && python download_model.py"
    exit 1
fi

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running BGE on $domain with hybrid extractive queries..."
    
    python scripts/baselines/retrieval_scripts/bge/bge_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --corpus_file "corpora/passage_level/${domain}.jsonl" \
        --query_file "scripts/ideas/retrieval_tasks/hybrid_extractive/datasets/${domain}_hybrid_extractive.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/hybrid_extractive/results/bge_${domain}_hybrid_extractive.jsonl" \
        --model_path "$BGE_MODEL_PATH" \
        --top_k 10 \
        --batch_size 64
    
    echo "✓ Completed: bge_${domain}_hybrid_extractive.jsonl"
    echo ""
done

echo "======================================"
echo "BGE Retrieval Complete!"
echo "======================================"
echo ""
echo "Results saved to: scripts/ideas/retrieval_tasks/hybrid_extractive/results/"
ls -lh scripts/ideas/retrieval_tasks/hybrid_extractive/results/bge_*.jsonl
echo ""

