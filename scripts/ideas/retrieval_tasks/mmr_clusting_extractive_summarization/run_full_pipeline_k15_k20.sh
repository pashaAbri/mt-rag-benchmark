#!/bin/bash
# Run full pipeline for k=15 and k=20: Rewriting -> Retrieval -> Evaluation

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

SCRIPT_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization"
K_VALUES=(15 20)

echo "============================================================"
echo "Full Pipeline: k=15 and k=20 Experiments"
echo "============================================================"
echo ""

for k in "${K_VALUES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing k=${k}"
    echo "============================================================"
    echo ""
    
    # Step 1: Run rewriting (if not already done)
    echo "Step 1: Running rewriting experiments (k=${k})..."
    echo "Note: Skipping rewriting step. Run run_rewrite_k15_k20.sh separately if needed."
    echo ""
    
    # Step 2: Run retrieval for all methods
    echo "Step 2: Running retrieval for all methods (k=${k})..."
    echo ""
    
    echo "  → BM25 retrieval..."
    "${SCRIPT_DIR}/run_bm25_mmr_k${k}.sh"
    echo ""
    
    echo "  → BGE retrieval..."
    "${SCRIPT_DIR}/run_bge_mmr_k${k}.sh"
    echo ""
    
    echo "  → ELSER retrieval..."
    "${SCRIPT_DIR}/run_elser_mmr_k${k}.sh"
    echo ""
    
    # Step 3: Run evaluation
    echo "Step 3: Running evaluation (k=${k})..."
    "${SCRIPT_DIR}/evaluate_all_mmr_k${k}.sh"
    echo ""
    
    echo "============================================================"
    echo "k=${k} pipeline complete!"
    echo "============================================================"
    echo ""
done

echo "============================================================"
echo "All experiments (k=15, k=20) complete!"
echo "============================================================"

