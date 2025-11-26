#!/bin/bash
# Run BM25 retrieval on MMR-rewritten datasets (k=10) for all domains

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

DOMAINS=("clapnq" "fiqa" "govt" "cloud")

echo "Running BM25 retrieval on MMR-rewritten datasets (k=10)..."
echo "Total experiments: ${#DOMAINS[@]}"
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running BM25 on $domain with MMR queries (k=10)..."
    
    # Use the full path to the python executable in the virtual environment just in case
    python scripts/baselines/retrieval_scripts/bm25/bm25_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --corpus_file "corpora/passage_level/${domain}.jsonl" \
        --query_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/datasets/${domain}_mmr_cluster_k10_lam0.7_all.jsonl" \
        --output_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k10/bm25_${domain}_mmr_cluster_k10.jsonl" \
        --top_k 10
    
    echo "✓ Completed: bm25_${domain}_mmr_cluster_k10.jsonl"
    echo ""
done

echo "All BM25 MMR (k=10) experiments complete!"
echo "Results saved to: scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k10/"

