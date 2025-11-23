#!/bin/bash

# Run cluster-based MMR rewriting for all domains

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR"

# Configuration
LAMBDA=0.7
NUM_SENTENCES=5
REPS_PER_CLUSTER=2

DOMAINS=("clapnq" "fiqa" "govt" "cloud")

echo "=========================================="
echo "Cluster-Based MMR Query Rewriting"
echo "Running for all domains"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Lambda: $LAMBDA"
echo "  MMR Sentences: $NUM_SENTENCES"
echo "  Reps per Cluster: $REPS_PER_CLUSTER"
echo "  Method: LLM-assisted rewriting (Mixtral)"
echo "  Output Dir: $OUTPUT_DIR"
echo ""
echo "=========================================="
echo ""

# Run for each domain
for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing domain: $domain"
    echo "=========================================="
    echo ""
    
    python3 "$SCRIPT_DIR/mmr_cluster_rewrite.py" \
        --domain "$domain" \
        --lambda-param "$LAMBDA" \
        --num-sentences "$NUM_SENTENCES" \
        --reps-per-cluster "$REPS_PER_CLUSTER" \
        --output-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully processed $domain"
        echo ""
    else
        echo ""
        echo "✗ Error processing $domain"
        echo ""
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All domains processed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR/datasets/"
echo ""
echo "Intermediate data:"
ls -lh "$OUTPUT_DIR/intermediate/"
echo ""

