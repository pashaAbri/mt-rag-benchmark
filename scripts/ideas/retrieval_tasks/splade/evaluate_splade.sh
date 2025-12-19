#!/bin/bash
# Evaluate all SPLADE retrieval results

set -e

# Navigate to project root
cd "$(dirname "$0")/../../../.."

# Activate venv
source .venv/bin/activate

RESULTS_DIR="scripts/ideas/retrieval_tasks/splade/results"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating SPLADE retrieval results..."
echo ""

# Remove any existing evaluation files first to ensure a clean run
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv

for result_file in "${RESULTS_DIR}"/splade_*.jsonl; do
    # Skip if no files found or if it's already an evaluated file
    [[ -e "$result_file" ]] || continue
    [[ "$result_file" == *"_evaluated"* ]] && continue
    
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
head -n 2 "${RESULTS_DIR}"/*_evaluated_aggregate.csv 2>/dev/null || echo "No aggregate files found"
