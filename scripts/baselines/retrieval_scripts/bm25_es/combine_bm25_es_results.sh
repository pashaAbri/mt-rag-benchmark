#!/bin/bash
# Combine BM25 ES retrieval results into 'all' domain files

set -e

RESULTS_DIR="scripts/baselines/retrieval_scripts/bm25_es/results"

echo "Combining results for 'all' domain..."
strategies=("lastturn" "questions" "rewrite")
domains=("clapnq" "cloud" "fiqa" "govt")

for strategy in "${strategies[@]}"; do
    output_file="${RESULTS_DIR}/bm25_es_all_${strategy}.jsonl"
    # Clear/Create the output file
    > "$output_file"
    
    echo "Creating ${output_file}..."
    for domain in "${domains[@]}"; do
        input_file="${RESULTS_DIR}/${domain}_${strategy}_results.jsonl"
        if [[ -f "$input_file" ]]; then
            # Use awk to append file content, ensuring a final newline is present
            # This prevents two JSON objects from ending up on the same line if the first file lacks a newline
            awk '1' "$input_file" >> "$output_file"
        else
            echo "Warning: $input_file not found, skipping for all_${strategy}"
        fi
    done
done

echo "Combination complete!"

