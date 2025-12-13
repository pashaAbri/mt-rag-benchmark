#!/bin/bash
# DH-RAG Experiment: Run query rewriting and retrieval across all domains

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "========================================"
echo "DH-RAG Experiment"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Configuration
ALPHA=${ALPHA:-0.6}
MAX_CLUSTERS=${MAX_CLUSTERS:-5}
CHAIN_THRESHOLD=${CHAIN_THRESHOLD:-0.4}
TOP_K=${TOP_K:-3}
DOMAINS=${DOMAINS:-"clapnq cloud fiqa govt"}
WORKERS=${WORKERS:-5}

echo "Configuration:"
echo "  Alpha: $ALPHA"
echo "  Max Clusters: $MAX_CLUSTERS"
echo "  Chain Threshold: $CHAIN_THRESHOLD"
echo "  Top-K: $TOP_K"
echo "  Domains: $DOMAINS"
echo "  Workers: $WORKERS"
echo ""

# Step 1: Run DH-RAG query rewriting
echo "========================================"
echo "Step 1: DH-RAG Query Rewriting"
echo "========================================"

python "$SCRIPT_DIR/run_dh_rag_rewrite.py" \
    --domains $DOMAINS \
    --alpha $ALPHA \
    --max_clusters $MAX_CLUSTERS \
    --chain_threshold $CHAIN_THRESHOLD \
    --top_k $TOP_K \
    --workers $WORKERS \
    --resume

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Run retrieval (reuse from targeted_rewrite)
echo "========================================"
echo "Step 2: Retrieval"
echo "========================================"

# Create symlink to retrieval script if needed
RETRIEVAL_SCRIPT="$SCRIPT_DIR/../targeted_rewrite/run_retrieval.py"
if [ -f "$RETRIEVAL_SCRIPT" ]; then
    for domain in $DOMAINS; do
        echo "Running retrieval for $domain..."
        
        # Use the DH-RAG rewritten queries
        INPUT_FILE="$SCRIPT_DIR/intermediate/dh_rag_${domain}.jsonl"
        
        if [ -f "$INPUT_FILE" ]; then
            for retriever in elser bge bm25; do
                OUTPUT_FILE="$SCRIPT_DIR/retrieval_results/dh_rag_${domain}_${retriever}.jsonl"
                
                if [ -f "$OUTPUT_FILE" ]; then
                    echo "  Skipping $retriever - output exists"
                    continue
                fi
                
                echo "  Running $retriever..."
                python "$RETRIEVAL_SCRIPT" \
                    --input_file "$INPUT_FILE" \
                    --output_file "$OUTPUT_FILE" \
                    --retriever "$retriever" \
                    --domain "$domain" \
                    --top_k 10
            done
        else
            echo "  Warning: $INPUT_FILE not found"
        fi
    done
else
    echo "Warning: Retrieval script not found at $RETRIEVAL_SCRIPT"
    echo "Please run retrieval manually"
fi

echo ""
echo "========================================"
echo "DH-RAG Experiment Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  Rewritten queries: $SCRIPT_DIR/intermediate/"
echo "  Retrieval results: $SCRIPT_DIR/retrieval_results/"

