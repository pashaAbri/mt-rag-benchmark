#!/bin/bash
# Combine ELSER MMR retrieval results into 'all' domain file

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results"

echo "Combining results for 'all' domain (ELSER MMR Cluster)..."
domains=("clapnq" "cloud" "fiqa" "govt")

output_file="${RESULTS_DIR}/elser_all_mmr_cluster.jsonl"
# Clear/Create the output file
> "$output_file"

echo "Creating ${output_file}..."
for domain in "${domains[@]}"; do
    input_file="${RESULTS_DIR}/elser_${domain}_mmr_cluster.jsonl"
    if [[ -f "$input_file" ]]; then
        # Use awk to append file content, ensuring a final newline is present
        awk '1' "$input_file" >> "$output_file"
    else
        echo "Warning: $input_file not found, skipping for all_mmr_cluster"
    fi
done

echo "Combination complete!"

