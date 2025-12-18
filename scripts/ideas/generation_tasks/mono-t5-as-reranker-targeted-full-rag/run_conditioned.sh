#!/bin/bash
# Run conditioned metrics (RL_F_idk, RB_llm_idk, RB_agg_idk) for models that need it
# Requires: idk_eval and RB_llm to be present

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

# Models that need conditioned metrics
FILES=(
    "gpt_4o_mono_t5_targeted_full_rag_evaluated.jsonl"
    "gpt_4o_mini_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_8b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_70b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "llama_3.1_405b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "qwen_2.5_7b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "qwen_2.5_72b_mono_t5_targeted_full_rag_evaluated.jsonl"
    "command_r_plus_mono_t5_targeted_full_rag_evaluated.jsonl"
    "mixtral_8x22b_mono_t5_targeted_full_rag_evaluated.jsonl"
)

echo "Running conditioned metrics evaluation..."
echo ""

for filename in "${FILES[@]}"; do
    file_path="${RESULTS_DIR}/${filename}"
    
    if [ -f "$file_path" ]; then
        echo "============================================"
        echo "Processing: $filename"
        echo "============================================"
        
        # Check prerequisites
        has_idk=$(head -1 "$file_path" | grep -c '"idk_eval"' || true)
        has_rb_llm=$(head -1 "$file_path" | grep -c '"RB_llm"' || true)
        
        if [ "$has_idk" -eq 0 ]; then
            echo "  Skipping - missing idk_eval (run idk_eval first)"
            echo ""
            continue
        fi
        
        if [ "$has_rb_llm" -eq 0 ]; then
            echo "  Skipping - missing RB_llm (run rb_llm first)"
            echo ""
            continue
        fi
        
        echo "  Computing conditioned metrics..."
        $PYTHON -c "
import sys
sys.path.insert(0, 'scripts/evaluation')
from judge_wrapper import get_idk_conditioned_metrics

print('  Calculating RL_F_idk, RB_llm_idk, RB_agg_idk...')
get_idk_conditioned_metrics('$file_path', '$file_path')
print('  Done.')
"
        echo ""
    else
        echo "Skipping $filename (not found)"
        echo ""
    fi
done

echo "============================================"
echo "Conditioned metrics complete!"
echo "============================================"
