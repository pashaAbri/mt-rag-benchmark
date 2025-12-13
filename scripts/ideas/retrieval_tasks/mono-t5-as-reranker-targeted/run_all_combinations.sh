#!/bin/bash
# Run all MonoT5 reranking combinations (Targeted Rewrite Edition) and evaluate results

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default to 'targeted_rewrite' query
RERANK_QUERY="${1:-targeted_rewrite}"

echo "================================================================================"
echo "Running All MonoT5 Reranking Combinations (Targeted Rewrite)"
echo "Reranking Query: $RERANK_QUERY"
echo "================================================================================"
echo ""

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Step 1: Run all combinations
echo "Step 1: Reranking all strategy combinations..."
echo "------------------------------------------------------------------------"
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/run_rerank.py" --rerank-query "$RERANK_QUERY"

echo ""
echo "================================================================================"
echo "Step 2: Evaluating all results and generating summaries..."
echo "================================================================================"
echo ""

# Step 2: Evaluate all results
python "$SCRIPT_DIR/summarize_evaluations.py" --rerank-query "$RERANK_QUERY"

echo ""
echo "================================================================================"
echo "All combinations completed successfully!"
echo "================================================================================"

