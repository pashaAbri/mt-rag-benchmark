#!/bin/bash
# Combine BM25 MMR retrieval results (k=15) into 'all' domain file

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k15"

echo "Combining results for 'all' domain (BM25 MMR Cluster k=15)..."
domains=("clapnq" "cloud" "fiqa" "govt")

output_file="${RESULTS_DIR}/bm25_all_mmr_cluster_k15.jsonl"
# Clear/Create the output file
> "$output_file"

echo "Creating ${output_file}..."
for domain in "${domains[@]}"; do
    input_file="${RESULTS_DIR}/bm25_${domain}_mmr_cluster_k15.jsonl"
    if [[ -f "$input_file" ]]; then
        awk '1' "$input_file" >> "$output_file"
    else
        echo "Warning: $input_file not found, skipping for all_mmr_cluster_k15"
    fi
done

echo "Combination complete!"

