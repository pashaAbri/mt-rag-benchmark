#!/bin/bash
# Run BM25 retrieval on MMR-rewritten datasets (lambda=0.85, k=10) for all domains

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
LAMBDA=0.85
K_VALUE=10

echo "Running BM25 retrieval on MMR-rewritten datasets (lambda=${LAMBDA}, k=${K_VALUE})..."
echo "Total experiments: ${#DOMAINS[@]}"
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

# Create results directory if it doesn't exist
RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_lam${LAMBDA}"
mkdir -p "${RESULTS_DIR}"

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running BM25 on $domain with MMR queries (lambda=${LAMBDA}, k=${K_VALUE})..."
    
    python scripts/baselines/retrieval_scripts/bm25/bm25_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --corpus_file "corpora/passage_level/${domain}.jsonl" \
        --query_file "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/datasets/${domain}_mmr_cluster_k${K_VALUE}_lam${LAMBDA}_all.jsonl" \
        --output_file "${RESULTS_DIR}/bm25_${domain}_mmr_cluster_k${K_VALUE}_lam${LAMBDA}.jsonl" \
        --top_k 10
    
    echo "✓ Completed: bm25_${domain}_mmr_cluster_k${K_VALUE}_lam${LAMBDA}.jsonl"
    echo ""
done

echo "All BM25 MMR (lambda=${LAMBDA}, k=${K_VALUE}) experiments complete!"
echo "Results saved to: ${RESULTS_DIR}/"

