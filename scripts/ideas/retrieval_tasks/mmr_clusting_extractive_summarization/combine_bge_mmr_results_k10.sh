#!/bin/bash
# Combine BGE MMR retrieval results (k=10) into 'all' domain file

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k10"

echo "Combining results for 'all' domain (BGE MMR Cluster k=10)..."
domains=("clapnq" "cloud" "fiqa" "govt")

output_file="${RESULTS_DIR}/bge_all_mmr_cluster_k10.jsonl"
# Clear/Create the output file
> "$output_file"

echo "Creating ${output_file}..."
for domain in "${domains[@]}"; do
    input_file="${RESULTS_DIR}/bge_${domain}_mmr_cluster_k10.jsonl"
    if [[ -f "$input_file" ]]; then
        # Use awk to append file content, ensuring a final newline is present
        awk '1' "$input_file" >> "$output_file"
    else
        echo "Warning: $input_file not found, skipping for all_mmr_cluster_k10"
    fi
done

echo "Combination complete!"

