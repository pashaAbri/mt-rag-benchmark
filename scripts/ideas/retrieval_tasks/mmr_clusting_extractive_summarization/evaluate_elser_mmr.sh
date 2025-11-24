#!/bin/bash
# Evaluate ELSER retrieval results for MMR-rewritten queries

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating ELSER retrieval results for MMR clustering..."
echo ""

# Remove any existing evaluation files first to ensure a clean run
# Using glob only for elser files
rm -f "${RESULTS_DIR}"/elser_*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/elser_*_evaluated_aggregate.csv

# Combine results first
echo "Combining results..."
chmod +x scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_elser_mmr_results.sh
./scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_elser_mmr_results.sh
echo ""

# Loop through result files
for result_file in "${RESULTS_DIR}"/elser_*_mmr_cluster.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
    echo ""
done

echo "All ELSER MMR evaluations complete!"
echo ""
echo "Aggregate results (head of each CSV):"
head -n 2 "${RESULTS_DIR}"/elser_*_evaluated_aggregate.csv

