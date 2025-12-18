#!/bin/bash
# Run remaining evaluations: RB_llm for models that need it, then conditioned metrics for all
# Saves after each step to avoid losing progress

set -e

# Navigate to project root
cd "$(dirname "$0")/../../../../"

# Use project virtual environment
PYTHON=".venv/bin/python"

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

RESULTS_DIR="scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag/results"

echo "============================================"
echo "Running remaining evaluations..."
echo "============================================"
echo ""

# Step 1: Run RB_llm for models that need it
NEED_RB_LLM=(
    "llama_3.1_405b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_8b_mono_t5_targeted_full_rag_evaluated.jsonl"
)

echo "Step 1: Running RB_llm for models that need it..."
echo ""

for filename in "${NEED_RB_LLM[@]}"; do
    file_path="${RESULTS_DIR}/${filename}"
    
    if [ -f "$file_path" ]; then
        # Check if RB_llm is already present
        if grep -q '"RB_llm"' "$file_path"; then
            echo "  $filename - RB_llm already present, skipping."
        else
            echo "  $filename - Running RB_llm..."
            $PYTHON -c "
import sys
sys.path.insert(0, 'scripts/evaluation')
from judge_wrapper import run_radbench_judge
run_radbench_judge('openai', '$file_path', '$file_path')
print('  Done.')
"
        fi
    else
        echo "  $filename - Not found, skipping."
    fi
done

echo ""
echo "============================================"
echo "Step 2: Running conditioned metrics for all models that need it..."
echo "============================================"
echo ""

# Models that need conditioned metrics recalculated
NEED_CONDITIONED=(
    "gpt_4o_mini_mono_t5_targeted_full_rag_evaluated.jsonl"
    "qwen_2.5_72b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "qwen_2.5_7b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_70b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_405b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_8b_mono_t5_targeted_full_rag_evaluated.jsonl"
)

for filename in "${NEED_CONDITIONED[@]}"; do
    file_path="${RESULTS_DIR}/${filename}"
    
    if [ -f "$file_path" ]; then
        echo "  $filename - Running conditioned metrics..."
        $PYTHON -c "
import sys
sys.path.insert(0, 'scripts/evaluation')
from judge_wrapper import get_idk_conditioned_metrics
get_idk_conditioned_metrics('$file_path', '$file_path')
print('  Done.')
"
    else
        echo "  $filename - Not found, skipping."
    fi
done

echo ""
echo "============================================"
echo "All remaining evaluations complete!"
echo "============================================"

# Show final status
echo ""
echo "Final metrics check:"
for f in ${RESULTS_DIR}/*_evaluated.jsonl; do
    echo "=== $(basename $f) ==="
    head -1 "$f" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); m = d.get('metrics', {}); print(f'  Metrics: {list(m.keys())}')"
done
