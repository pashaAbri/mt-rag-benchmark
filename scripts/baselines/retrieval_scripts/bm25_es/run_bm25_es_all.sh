#!/bin/bash
# Run BM25 Elasticsearch retrieval for all domains and query types

# ==================================================================================================
# SETUP INSTRUCTIONS
# ==================================================================================================
# 1. Elasticsearch Installation:
#    Ensure you have Elasticsearch 8.11.0 installed in the project root.
#    Location: ./elasticsearch-8.11.0
#
# 2. Starting Elasticsearch:
#    The script will attempt to start Elasticsearch if it's not running.
#    It runs in single-node mode with security disabled for local testing.
#    Command: ./elasticsearch-8.11.0/bin/elasticsearch -d -E xpack.security.enabled=false -E discovery.type=single-node
#
# 3. Python Environment:
#    Ensure your virtual environment is activated and dependencies are installed.
#    - Activate: source .venv/bin/activate (or similar)
#    - Install: pip install "elasticsearch<9.0.0" tqdm
#      (Note: Use elasticsearch client version 8.x to match server version 8.11.0)
# ==================================================================================================

set -e  # Exit on error

# Elasticsearch configuration
ES_HOST="localhost"
ES_PORT=9200
ES_DIR="elasticsearch-8.11.0"

# Function to check if Elasticsearch is running
check_es() {
    curl -s "http://${ES_HOST}:${ES_PORT}" > /dev/null
    return $?
}

# Start Elasticsearch if not running
if ! check_es; then
    echo "Elasticsearch is not running. Attempting to start..."
    
    if [ -d "$ES_DIR" ]; then
        echo "Starting Elasticsearch from $ES_DIR..."
        ./$ES_DIR/bin/elasticsearch -d -E xpack.security.enabled=false -E discovery.type=single-node
        
        echo "Waiting for Elasticsearch to start (this may take a few moments)..."
        # Wait loop
        for i in {1..60}; do
            if check_es; then
                echo "Elasticsearch started successfully!"
                break
            fi
            echo -n "."
            sleep 1
        done
        
        if ! check_es; then
            echo ""
            echo "Error: Elasticsearch failed to start within 60 seconds."
            echo "Check logs at: $ES_DIR/logs/elasticsearch.log"
            exit 1
        fi
        echo ""
    else
        echo "Error: Elasticsearch directory '$ES_DIR' not found."
        echo "Please download and extract Elasticsearch 8.11.0 to the project root."
        exit 1
    fi
else
    echo "Elasticsearch is already running."
fi

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
QUERY_TYPES=("lastturn" "rewrite" "questions")

RESULTS_DIR="scripts/baselines/retrieval_scripts/bm25_es/results"
mkdir -p "$RESULTS_DIR"

echo "Running BM25 ES retrieval on all domains and query types..."
echo "Total experiments: $((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))"
echo ""

COUNT=0
TOTAL=$((${#DOMAINS[@]} * ${#QUERY_TYPES[@]}))

for domain in "${DOMAINS[@]}"; do
    for query_type in "${QUERY_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running BM25 ES on $domain with $query_type queries..."
        
        python scripts/baselines/retrieval_scripts/bm25_es/bm25_retrieval.py \
            --domain "$domain" \
            --query_type "$query_type" \
            --corpus_file "corpora/passage_level/${domain}.jsonl" \
            --query_file "human/retrieval_tasks/${domain}/${domain}_${query_type}.jsonl" \
            --output_file "${RESULTS_DIR}/${domain}_${query_type}_results.jsonl" \
            --top_k 10 \
            --es_host "$ES_HOST" \
            --es_port "$ES_PORT"
        
        echo "✓ Completed: ${domain}_${query_type}_results.jsonl"
        echo ""
    done
done

echo "All BM25 ES experiments complete!"
echo "Results saved to: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}/"*_results.jsonl
