#!/bin/bash
# Evaluate BM25 retrieval results for MMR-rewritten queries (k=10)

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k10"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating BM25 retrieval results for MMR clustering (k=10)..."
echo ""

# Remove any existing evaluation files first to ensure a clean run
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv

# Combine results first
echo "Combining results..."
chmod +x scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_bm25_mmr_results_k10.sh
./scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_bm25_mmr_results_k10.sh
echo ""

# Loop through result files
for result_file in "${RESULTS_DIR}"/bm25_*_mmr_cluster_k10.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
    echo ""
done

echo "All MMR (k=10) evaluations complete!"
echo ""
echo "Aggregate results (head of each CSV):"
head -n 2 "${RESULTS_DIR}"/*_evaluated_aggregate.csv

