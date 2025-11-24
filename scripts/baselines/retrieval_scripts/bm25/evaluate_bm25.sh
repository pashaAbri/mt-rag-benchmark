#!/bin/bash
# Evaluate all BM25 retrieval results

set -e

RESULTS_DIR="scripts/baselines/retrieval_scripts/bm25/results"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating BM25 retrieval results..."
echo ""

# Loop through all jsonl files in the results directory
# Remove any existing evaluation files first to ensure a clean run
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv

for result_file in "${RESULTS_DIR}"/bm25_*.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
    echo ""
done

echo "All evaluations complete!"
echo ""
echo "Aggregate results (head of each CSV):"
# Just show the header and first few lines to verify
head -n 2 "${RESULTS_DIR}"/*_evaluated_aggregate.csv
