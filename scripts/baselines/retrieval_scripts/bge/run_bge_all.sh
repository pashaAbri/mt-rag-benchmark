#!/bin/bash
# Run BGE retrieval on all domains and query types

set -e  # Exit on error

# Navigate to project root
cd "$(dirname "$0")/../../.."

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

echo "Running BGE retrieval on all domains and query types..."
echo "Total experiments: $((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))"
echo ""

COUNT=0
TOTAL=$((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))

for domain in "${DOMAINS[@]}"; do
    for query_type in "${QUERY_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running BGE on $domain with $query_type queries..."
        
        python scripts/baselines/retrieval_scripts/bge/bge_retrieval.py \
            --domain "$domain" \
            --query_type "$query_type" \
            --corpus_file "corpora/passage_level/${domain}.jsonl" \
            --query_file "human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
            --output_file "scripts/baselines/retrieval_scripts/bge/results/bge_${domain}_${query_type}.jsonl" \
            --top_k 10
        
        echo "✓ Completed: bge_${domain}_${query_type}.jsonl"
        echo ""
    done
done

echo "All BGE experiments complete!"
echo "Results saved to: scripts/baselines/retrieval_scripts/bge/results/"
ls -lh scripts/baselines/retrieval_scripts/bge/results/bge_*.jsonl


