#!/bin/bash
# Run RAGAS evaluation for Mono-T5 Targeted Full RAG results

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

# List of files to evaluate
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

for filename in "${FILES[@]}"; do
    file_path="${RESULTS_DIR}/${filename}"
    
    if [ -f "$file_path" ]; then
        echo "Processing $filename..."
        
        # Check if RAGAS (RL_F) is already present
        if grep -q "RL_F" "$file_path"; then
            echo "  RAGAS already evaluated (RL_F found)."
        else
            echo "  Running RAGAS evaluation..."
            $PYTHON scripts/evaluation/run_ragas_only.py \
                -i "$file_path" \
                -o "$file_path"
            echo "  Done."
        fi
    else
        echo "Skipping $filename (not found)"
    fi
    echo ""
done
