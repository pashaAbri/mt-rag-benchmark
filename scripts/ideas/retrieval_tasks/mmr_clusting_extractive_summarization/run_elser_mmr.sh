#!/bin/bash
# Run ELSER retrieval on MMR-rewritten datasets for all domains

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

# Load environment variables from .env if present (though python script does this too)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

DOMAINS=("clapnq" "fiqa" "govt" "cloud")

echo "Running ELSER retrieval on MMR-rewritten datasets..."
echo "Total experiments: ${#DOMAINS[@]}"
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running ELSER on $domain with MMR queries..."
    
    python scripts/baselines/retrieval_scripts/elser/elser_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --query_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/datasets/${domain}_mmr_cluster_all.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results/elser_${domain}_mmr_cluster.jsonl" \
        --top_k 10 \
        --delay 0.2  # Slightly faster delay as we have many queries
    
    echo "✓ Completed: elser_${domain}_mmr_cluster.jsonl"
    echo ""
done

echo "All ELSER MMR experiments complete!"
echo "Results saved to: scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results/"

