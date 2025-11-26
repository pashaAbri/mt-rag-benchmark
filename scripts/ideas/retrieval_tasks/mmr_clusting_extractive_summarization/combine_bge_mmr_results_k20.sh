#!/bin/bash
# Combine BGE MMR retrieval results (k=20) into 'all' domain file

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k20"

echo "Combining results for 'all' domain (BGE MMR Cluster k=20)..."
domains=("clapnq" "cloud" "fiqa" "govt")

output_file="${RESULTS_DIR}/bge_all_mmr_cluster_k20.jsonl"
> "$output_file"

echo "Creating ${output_file}..."
for domain in "${domains[@]}"; do
    input_file="${RESULTS_DIR}/bge_${domain}_mmr_cluster_k20.jsonl"
    if [[ -f "$input_file" ]]; then
        awk '1' "$input_file" >> "$output_file"
    else
        echo "Warning: $input_file not found, skipping for all_mmr_cluster_k20"
    fi
done

echo "Combination complete!"

