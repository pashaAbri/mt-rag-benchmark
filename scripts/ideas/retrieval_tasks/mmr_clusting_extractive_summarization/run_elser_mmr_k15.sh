#!/bin/bash
# Run ELSER retrieval on MMR-rewritten datasets (k=15) for all domains

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

# Load environment variables from .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

DOMAINS=("clapnq" "fiqa" "govt" "cloud")

echo "Running ELSER retrieval on MMR-rewritten datasets (k=15)..."
echo "Total experiments: ${#DOMAINS[@]}"
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running ELSER on $domain with MMR queries (k=15)..."
    
    python scripts/baselines/retrieval_scripts/elser/elser_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --query_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/datasets/${domain}_mmr_cluster_k15_lam0.7_all.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k15/elser_${domain}_mmr_cluster_k15.jsonl" \
        --top_k 10 \
        --delay 0.2
    
    echo "✓ Completed: elser_${domain}_mmr_cluster_k15.jsonl"
    echo ""
done

echo "All ELSER MMR (k=15) experiments complete!"
echo "Results saved to: scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k15/"

