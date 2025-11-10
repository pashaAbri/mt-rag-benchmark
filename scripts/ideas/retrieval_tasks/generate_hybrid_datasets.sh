#!/bin/bash
# Generate hybrid extractive datasets for MT-RAG benchmark

set -e

cd "$(dirname "$0")"

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "======================================"
echo "Generating Hybrid Extractive Datasets"
echo "======================================"
echo ""

echo "Step 1: Checking/Downloading Model"
echo "--------------------------------------"
python download_model.py
echo ""

echo "Step 2: Running Hybrid Extractive Rewriting..."
echo "--------------------------------------"
for domain in "${DOMAINS[@]}"; do
    echo "Processing $domain with Hybrid Extractive..."
    python hybrid_extractive/hybrid_extractive_rewrite.py "$domain"
    echo ""
done

echo "Hybrid Extractive complete!"
echo ""

echo "======================================"
echo "Dataset Generation Complete!"
echo "======================================"
echo ""
echo "Generated datasets:"
echo ""
for domain in "${DOMAINS[@]}"; do
    echo "  - hybrid_extractive/datasets/${domain}_hybrid_extractive.jsonl"
done
echo ""

