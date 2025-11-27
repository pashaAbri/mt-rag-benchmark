#!/bin/bash
# Run full pipeline for lambda experiments: Rewriting -> Retrieval -> Evaluation
# Tests lambda values: 0.3, 0.5, 0.85, 0.9 (with k=15 fixed)

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e  # Exit on error

SCRIPT_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization"
LAMBDA_VALUES=(0.3 0.5 0.85 0.9)

echo "============================================================"
echo "Full Pipeline: Lambda Parameter Experiments"
echo "Testing lambda values: ${LAMBDA_VALUES[@]}"
echo "Fixed k value: 10"
echo "============================================================"
echo ""

for lambda in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing lambda=${lambda}"
    echo "============================================================"
    echo ""
    
    # Step 1: Run rewriting (if not already done)
    echo "Step 1: Running rewriting experiments (lambda=${lambda})..."
    echo "Note: Skipping rewriting step. Run run_rewrite_lambda.sh separately if needed."
    echo ""
    
    # Step 2: Run retrieval for all methods
    echo "Step 2: Running retrieval for all methods (lambda=${lambda})..."
    echo ""
    
    echo "  → BM25 retrieval..."
    "${SCRIPT_DIR}/run_bm25_mmr_lam${lambda}.sh"
    echo ""
    
    echo "  → BGE retrieval..."
    "${SCRIPT_DIR}/run_bge_mmr_lam${lambda}.sh"
    echo ""
    
    echo "  → ELSER retrieval..."
    "${SCRIPT_DIR}/run_elser_mmr_lam${lambda}.sh"
    echo ""
    
    # Step 3: Run evaluation
    echo "Step 3: Running evaluation (lambda=${lambda})..."
    "${SCRIPT_DIR}/evaluate_all_mmr_lam${lambda}.sh"
    echo ""
    
    echo "============================================================"
    echo "lambda=${lambda} pipeline complete!"
    echo "============================================================"
    echo ""
done

echo "============================================================"
echo "All lambda experiments complete!"
echo "============================================================"
echo ""
echo "Results saved in:"
for lambda in "${LAMBDA_VALUES[@]}"; do
    echo "  - results_lam${lambda}/"
done

