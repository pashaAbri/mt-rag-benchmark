#!/bin/bash
# Evaluate all BM25 Elasticsearch retrieval results

set -e

RESULTS_DIR="scripts/baselines/retrieval_scripts/bm25_es/results"
SCRIPT_DIR="scripts/evaluation"

echo "Evaluating BM25 Elasticsearch retrieval results..."
echo ""

# Remove any existing evaluation files first to ensure a clean run
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv

# Combine results first
echo "Combining results..."
./scripts/baselines/retrieval_scripts/bm25_es/combine_bm25_es_results.sh
echo ""

# Loop through result files
for result_file in "${RESULTS_DIR}"/*.jsonl; do
    filename=$(basename "$result_file")
    
    # Filter for valid result files: either ending in _results.jsonl or starting with bm25_es_all_
    if [[ "$filename" =~ _results\.jsonl$ ]] || [[ "$filename" =~ ^bm25_es_all_.*\.jsonl$ ]]; then
        base_name="${filename%.jsonl}"
        # Remove '_results' suffix if present for cleaner output name
        base_name="${base_name%_results}"
        
        output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
        
        echo "Evaluating: $base_name"
        python "${SCRIPT_DIR}/run_retrieval_eval.py" \
            --input_file "$result_file" \
            --output_file "$output_file"
        echo ""
    fi
done

echo "All evaluations complete!"
echo ""
echo "Aggregate results (head of each CSV):"
head -n 2 "${RESULTS_DIR}"/*_evaluated_aggregate.csv
