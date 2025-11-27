#!/bin/bash
# Combine ELSER MMR retrieval results (lambda=0.9) into 'all' domain file

set -e

LAMBDA=0.9
K_VALUE=10
RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_lam${LAMBDA}"

echo "Combining results for 'all' domain (ELSER MMR Cluster lambda=${LAMBDA}, k=${K_VALUE})..."
domains=("clapnq" "cloud" "fiqa" "govt")

output_file="${RESULTS_DIR}/elser_all_mmr_cluster_k${K_VALUE}_lam${LAMBDA}.jsonl"
> "$output_file"

echo "Creating ${output_file}..."
for domain in "${domains[@]}"; do
    input_file="${RESULTS_DIR}/elser_${domain}_mmr_cluster_k${K_VALUE}_lam${LAMBDA}.jsonl"
    if [[ -f "$input_file" ]]; then
        awk '1' "$input_file" >> "$output_file"
    else
        echo "Warning: $input_file not found, skipping for all_mmr_cluster_k${K_VALUE}_lam${LAMBDA}"
    fi
done

echo "Combination complete!"

