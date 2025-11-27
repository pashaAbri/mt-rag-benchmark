#!/bin/bash
# Run MMR Cluster rewriting experiments for different lambda values
# Uses k=15 as baseline, varies lambda parameter

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

DOMAINS=("clapnq" "fiqa" "govt" "cloud")
LAMBDA_VALUES=(0.3 0.5 0.85 0.9)
K_VALUE=10  # Fixed k value for lambda experiments

SCRIPT_PATH="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/mmr_cluster_rewrite.py"

for lambda in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running MMR Cluster Rewriting for lambda=${lambda} (k=${K_VALUE})"
    echo "============================================================"
    echo ""
    
    for domain in "${DOMAINS[@]}"; do
        echo "[${domain}] Running rewriting with lambda=${lambda}, k=${K_VALUE}..."
        
        python3 "${SCRIPT_PATH}" \
            --domain "${domain}" \
            --num-sentences "${K_VALUE}" \
            --lambda-param "${lambda}" \
            --reps-per-cluster 3
        
        echo "✓ Completed: ${domain} (lambda=${lambda}, k=${K_VALUE})"
        echo ""
    done
    
    echo "============================================================"
    echo "lambda=${lambda} rewriting complete!"
    echo "============================================================"
    echo ""
done

echo "All rewriting experiments (lambda sweep) complete!"

