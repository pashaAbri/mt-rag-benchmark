#!/bin/bash
# Generate pure extractive rewrite datasets for MT-RAG benchmark
# This creates both analysis and MT-RAG formatted datasets for retrieval testing

set -e

cd "$(dirname "$0")"

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "======================================"
echo "Generating Pure Extractive Datasets"
echo "======================================"
echo ""

echo "Step 1: Checking/Downloading Model"
echo "--------------------------------------"
python download_model.py
echo ""

echo "Step 2: Running Pure Extractive Rewriting..."
echo "--------------------------------------"
for domain in "${DOMAINS[@]}"; do
    echo "Processing $domain with Pure Extractive..."
    python pure_extractive/pure_extractive_rewrite.py "$domain"
    echo ""
done

echo "Pure Extractive complete!"
echo ""

echo "======================================"
echo "Dataset Generation Complete!"
echo "======================================"
echo ""
echo "Generated datasets:"
echo ""
for domain in "${DOMAINS[@]}"; do
    echo "  - pure_extractive/datasets/${domain}_pure_extractive.jsonl"
done
echo ""
echo "Note: Hybrid Extractive can be run separately when needed."

