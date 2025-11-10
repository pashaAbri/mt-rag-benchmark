#!/bin/bash
# Apply RRF fusion to BGE retrieval results (lastturn + rewrite) - 2-way

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "=========================================="
echo "RRF Fusion: BGE (lastturn + rewrite) [2-way]"
echo "=========================================="
echo ""

COUNT=0
TOTAL=${#DOMAINS[@]}

for domain in "${DOMAINS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Fusing BGE results for $domain..."
    
    # Use the same collection names as baseline scripts
    case "$domain" in
        "clapnq") collection_name="mt-rag-clapnq-elser-512-100-20240503";;
        "govt") collection_name="mt-rag-govt-elser-512-100-20240611";;
        "fiqa") collection_name="mt-rag-fiqa-beir-elser-512-100-20240501";;
        "cloud") collection_name="mt-rag-ibmcloud-elser-512-100-20240502";;
    esac
    
    python scripts/ideas/retrieval_tasks/multi_strategy_fusion/rrf_fusion.py \
        --input_files \
            scripts/baselines/retrieval_scripts/bge/results/bge_${domain}_lastturn.jsonl \
            scripts/baselines/retrieval_scripts/bge/results/bge_${domain}_rewrite.jsonl \
        --output_file scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/datasets/bge_${domain}_fusion_2way.jsonl \
        --collection_name "$collection_name" \
        --top_k 10 \
        --rrf_k 60
    
    echo ""
done

echo "=========================================="
echo "✓ Fusion Complete!"
echo "=========================================="
echo ""
echo "Datasets saved to: scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/datasets/"
ls -lh scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/datasets/bge_*_fusion_2way.jsonl
echo ""

