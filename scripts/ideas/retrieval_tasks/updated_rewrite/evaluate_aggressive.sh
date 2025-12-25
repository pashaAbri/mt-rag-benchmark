#!/bin/bash
# Evaluate Aggressive Rewrite retrieval results

RESULTS_DIR="scripts/ideas/retrieval_tasks/updated_rewrite/results"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating Aggressive Rewrite retrieval results..."
echo ""

# Remove any existing evaluation files
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv

# Evaluate each result file
for result_file in "${RESULTS_DIR}"/elser_*_aggressive_rewrite.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    /Users/pastil/Dev/Github/mt-rag-benchmark/.venv/bin/python3 "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
    echo ""
done

echo "All evaluations complete!"
echo ""
echo "Aggregate results:"
cat "${RESULTS_DIR}"/*_evaluated_aggregate.csv

