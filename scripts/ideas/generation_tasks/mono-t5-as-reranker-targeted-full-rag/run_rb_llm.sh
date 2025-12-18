#!/bin/bash
# Run RB_llm (RADBench LLM Judge) for models that need it
# With checkpointing to save progress every 50 rows

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

# Models that need RB_llm
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

echo "Running RB_llm evaluation (with checkpointing)..."
echo ""

for filename in "${FILES[@]}"; do
    file_path="${RESULTS_DIR}/${filename}"
    
    if [ -f "$file_path" ]; then
        echo "============================================"
        echo "Processing: $filename"
        echo "============================================"
        
        # Check if RB_llm is already complete (check last row)
        last_has_rb_llm=$(tail -1 "$file_path" | grep -c '"RB_llm"' || true)
        first_has_rb_llm=$(head -1 "$file_path" | grep -c '"RB_llm"' || true)
        
        if [ "$first_has_rb_llm" -eq 1 ] && [ "$last_has_rb_llm" -eq 1 ]; then
            echo "  Skipping - RB_llm already complete."
            echo ""
            continue
        fi
        
        echo "  Running RB_llm with checkpointing..."
        $PYTHON scripts/evaluation/run_rb_llm_only.py \
            -i "$file_path" \
            -o "$file_path" \
            --save-every 50
        
        echo ""
    else
        echo "Skipping $filename (not found)"
        echo ""
    fi
done

echo "============================================"
echo "RB_llm evaluation complete!"
echo "============================================"
