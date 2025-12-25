#!/bin/bash
# Evaluate Aggressive Rewrite retrieval results
# Usage: bash evaluate_aggressive.sh [prompt_name]

PROMPT_NAME=${1:-aggressive}
RESULTS_DIR="scripts/ideas/retrieval_tasks/updated_rewrite/results/${PROMPT_NAME}"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating Retrieval Results for Strategy: ${PROMPT_NAME}"
echo "Results Directory: ${RESULTS_DIR}"
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory not found: $RESULTS_DIR"
    exit 1
fi

# Remove any existing evaluation files
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv

# Evaluate each result file
# Match pattern: elser_{domain}_{prompt_name}_rewrite.jsonl
for result_file in "${RESULTS_DIR}"/elser_*_"${PROMPT_NAME}"_rewrite.jsonl; do
    if [ ! -f "$result_file" ]; then
        echo "No results found matching pattern: ${RESULTS_DIR}/elser_*_${PROMPT_NAME}_rewrite.jsonl"
        continue
    fi

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

