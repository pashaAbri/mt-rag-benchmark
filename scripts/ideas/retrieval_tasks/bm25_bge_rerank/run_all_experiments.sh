#!/bin/bash
# Run all two-stage retrieval experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Two-Stage BM25 + BGE Reranking Experiments"
echo "=============================================="

# Activate virtual environment if it exists
if [ -f "../../../../venv/bin/activate" ]; then
    source "../../../../venv/bin/activate"
    echo "Activated virtual environment"
fi

# Run all combinations
echo ""
echo "Running two-stage retrieval for all domains and query types..."
python run_two_stage_retrieval.py

# Compare with baselines
echo ""
echo "Comparing results with baselines..."
python compare_with_baselines.py

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Results saved in:"
echo "  - results/bm25_bge_rerank_*_evaluated.jsonl"
echo "  - results/experiment_summary.csv"
echo "  - results/comparison_with_baselines.csv"

