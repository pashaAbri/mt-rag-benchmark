#!/bin/bash
# Run BGE retrieval on MMR-rewritten datasets for all domains

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

DOMAINS=("clapnq" "fiqa" "govt" "cloud")

echo "Running BGE retrieval on MMR-rewritten datasets..."
echo "Total experiments: ${#DOMAINS[@]}"
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running BGE on $domain with MMR queries..."
    
    python scripts/baselines/retrieval_scripts/bge/bge_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --corpus_file "corpora/passage_level/${domain}.jsonl" \
        --query_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/datasets/${domain}_mmr_cluster_all.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results/bge_${domain}_mmr_cluster.jsonl" \
        --top_k 10
    
    echo "✓ Completed: bge_${domain}_mmr_cluster.jsonl"
    echo ""
done

echo "All BGE MMR experiments complete!"
echo "Results saved to: scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results/"

