#!/bin/bash
# Run SPLADE retrieval on all domains and query types

set -e  # Exit on error

# Navigate to project root
cd "$(dirname "$0")/../../../.."

# Activate venv
source .venv/bin/activate

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

echo "Running SPLADE retrieval on all domains and query types..."
echo "Total experiments: $((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))"
echo ""

COUNT=0
TOTAL=$((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))

for domain in "${DOMAINS[@]}"; do
    for query_type in "${QUERY_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running SPLADE on $domain with $query_type queries..."
        
        python scripts/ideas/retrieval_tasks/splade/splade_retrieval.py \
            --domain "$domain" \
            --query_type "$query_type" \
            --corpus_file "corpora/passage_level/${domain}.jsonl" \
            --query_file "human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
            --output_file "scripts/ideas/retrieval_tasks/splade/results/splade_${domain}_${query_type}.jsonl" \
            --top_k 10
        
        echo "✓ Completed: splade_${domain}_${query_type}.jsonl"
        echo ""
    done
done

echo "All SPLADE experiments complete!"
echo "Results saved to: scripts/ideas/retrieval_tasks/splade/results/"
ls -lh scripts/ideas/retrieval_tasks/splade/results/splade_*.jsonl
