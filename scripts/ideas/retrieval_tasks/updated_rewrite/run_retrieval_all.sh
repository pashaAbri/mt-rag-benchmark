#!/bin/bash
# Run ELSER retrieval with Aggressive Rewrite queries

# Ensure env vars are set
if [ -z "$ES_URL" ]; then
    export ES_URL=$(grep ES_URL .env | cut -d '=' -f2)
fi
if [ -z "$ES_API_KEY" ]; then
    export ES_API_KEY=$(grep ES_API_KEY .env | cut -d '=' -f2)
fi

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_FILE_DIR="scripts/ideas/retrieval_tasks/updated_rewrite/results"
OUTPUT_DIR="scripts/ideas/retrieval_tasks/updated_rewrite/results"

mkdir -p "$OUTPUT_DIR"

for domain in "${DOMAINS[@]}"; do
    echo "Running ELSER on $domain with Aggressive Rewrite..."
    
    # Check if query file exists
    QUERY_FILE="${QUERY_FILE_DIR}/${domain}_aggressive_rewrite.jsonl"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "Error: Query file not found: $QUERY_FILE"
        echo "Please run run_aggressive_rewrite.py first."
        continue
    fi

    # Run retrieval
    # We use 'rewrite' as query_type to satisfy the argument validation, 
    # but we provide our custom file.
    /Users/pastil/Dev/Github/mt-rag-benchmark/.venv/bin/python3 scripts/baselines/retrieval_scripts/elser/elser_retrieval.py \
        --domain "$domain" \
        --query_type "rewrite" \
        --query_file "$QUERY_FILE" \
        --output_file "${OUTPUT_DIR}/elser_${domain}_aggressive_rewrite.jsonl" \
        --top_k 10 \
        --delay 0.5
    
    echo "Finished $domain"
    echo "------------------------------------------------"
done

