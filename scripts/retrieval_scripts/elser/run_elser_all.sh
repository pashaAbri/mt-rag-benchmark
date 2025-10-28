#!/bin/bash
# Run ELSER retrieval on all working domains and query types
# Note: ClapNQ is excluded as it's currently being reindexed

set -e  # Exit on error

# Only include domains that have ELSER fully indexed
DOMAINS=("fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

echo "Running ELSER retrieval on all working domains and query types..."
echo "Note: ClapNQ excluded (currently reindexing)"
echo "Total experiments: $((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))"
echo ""

COUNT=0
TOTAL=$((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))

for domain in "${DOMAINS[@]}"; do
    for query_type in "${QUERY_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running ELSER on $domain with $query_type queries..."
        
        python scripts/retrieval_scripts/elser/elser_retrieval.py \
            --domain "$domain" \
            --query_type "$query_type" \
            --query_file "human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
            --output_file "scripts/retrieval_scripts/elser/results/elser_${domain}_${query_type}.jsonl" \
            --top_k 10
        
        echo "✓ Completed: elser_${domain}_${query_type}.jsonl"
        echo ""
    done
done

echo "All ELSER experiments complete!"
echo "Results saved to: scripts/retrieval_scripts/elser/results/"
ls -lh scripts/retrieval_scripts/elser/results/elser_*.jsonl

echo ""
echo "Note: ClapNQ will be available once reindexing completes (~12-15 hours)"
echo "Check progress: python -c \"from elasticsearch import Elasticsearch; import os; es = Elasticsearch(os.getenv('ES_URL'), api_key=os.getenv('ES_API_KEY')); print(f'Progress: {es.count(index=\\\"mtrag-clapnq-elser-512-100-reindexed\\\")[\\\"count\\\"]:,} / 183,408')\""

