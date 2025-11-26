#!/bin/bash
# Run MMR Cluster rewriting experiments for k=15 and k=20

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
K_VALUES=(15 20)

SCRIPT_PATH="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/mmr_cluster_rewrite.py"

for k in "${K_VALUES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running MMR Cluster Rewriting for k=${k}"
    echo "============================================================"
    echo ""
    
    for domain in "${DOMAINS[@]}"; do
        echo "[${domain}] Running rewriting with k=${k}..."
        
        python3 "${SCRIPT_PATH}" \
            --domain "${domain}" \
            --num-sentences "${k}" \
            --lambda-param 0.7 \
            --reps-per-cluster 3
        
        echo "✓ Completed: ${domain} (k=${k})"
        echo ""
    done
    
    echo "============================================================"
    echo "k=${k} rewriting complete!"
    echo "============================================================"
    echo ""
done

echo "All rewriting experiments (k=15, k=20) complete!"

