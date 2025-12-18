#!/bin/bash
# Run IDK evaluation for Mono-T5 Targeted Full RAG results
# Saves intermittently to avoid losing progress

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

# Models that need idk_eval
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

echo "Running IDK evaluation for Mono-T5 Targeted Full RAG..."
echo ""

for filename in "${FILES[@]}"; do
    file_path="${RESULTS_DIR}/${filename}"
    
    if [ -f "$file_path" ]; then
        echo "============================================"
        echo "Processing: $filename"
        echo "============================================"
        
        # Check if idk_eval is already present
        if grep -q '"idk_eval"' "$file_path"; then
            # Check if ALL rows have idk_eval (sample first and last row)
            first_has_idk=$(head -1 "$file_path" | grep -c '"idk_eval"' || true)
            last_has_idk=$(tail -1 "$file_path" | grep -c '"idk_eval"' || true)
            
            if [ "$first_has_idk" -eq 1 ] && [ "$last_has_idk" -eq 1 ]; then
                echo "  Skipping - idk_eval already complete."
                echo ""
                continue
            else
                echo "  Resuming partial idk_eval..."
            fi
        else
            echo "  Running idk_eval from scratch..."
        fi
        
        $PYTHON scripts/evaluation/run_idk_only.py \
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
echo "IDK evaluation complete!"
echo "============================================"
