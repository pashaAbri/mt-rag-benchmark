#!/bin/bash
# Run ELSER retrieval on all domains and query types

set -e  # Exit on error

# All 4 domains with ELSER fully indexed
DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

echo "Running ELSER retrieval on all domains and query types..."
echo "Total experiments: $((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))"
echo ""

COUNT=0
TOTAL=$((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))

for domain in "${DOMAINS[@]}"; do
    for query_type in "${QUERY_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running ELSER on $domain with $query_type queries..."
        
        python scripts/baselines/retrieval_scripts/elser/elser_retrieval.py \
            --domain "$domain" \
            --query_type "$query_type" \
            --query_file "human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
            --output_file "scripts/baselines/retrieval_scripts/elser/results/elser_${domain}_${query_type}.jsonl" \
            --top_k 10 \
            --delay 2.0
        
        echo "✓ Completed: elser_${domain}_${query_type}.jsonl"
        echo ""
    done
done

echo "All ELSER experiments complete!"
echo "Results saved to: scripts/baselines/retrieval_scripts/elser/results/"
echo ""
echo "Summary:"
wc -l scripts/baselines/retrieval_scripts/elser/results/elser_*.jsonl | tail -1
echo ""
echo "Run evaluation with:"
echo "  bash scripts/baselines/retrieval_scripts/elser/evaluate_elser.sh"

