#!/bin/bash
# Run BM25 retrieval on all domains and query types

set -e  # Exit on error

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

echo "Running BM25 retrieval on all domains and query types..."
echo "Total experiments: $((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))"
echo ""

COUNT=0
TOTAL=$((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))

for domain in "${DOMAINS[@]}"; do
    for query_type in "${QUERY_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running BM25 on $domain with $query_type queries..."
        
        python scripts/baselines/retrieval_scripts/bm25/bm25_retrieval.py \
            --domain "$domain" \
            --query_type "$query_type" \
            --corpus_file "corpora/passage_level/${domain}.jsonl" \
            --query_file "human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
            --output_file "scripts/baselines/retrieval_scripts/bm25/results/bm25_${domain}_${query_type}.jsonl" \
            --top_k 10 2>&1 | grep -E "(Loaded|Building|built|Running|complete|saved)"
        
        echo "✓ Completed: bm25_${domain}_${query_type}.jsonl"
        echo ""
    done
done

echo "All BM25 experiments complete!"
echo "Results saved to: scripts/baselines/retrieval_scripts/bm25/results/"
ls -lh scripts/baselines/retrieval_scripts/bm25/results/bm25_*.jsonl

