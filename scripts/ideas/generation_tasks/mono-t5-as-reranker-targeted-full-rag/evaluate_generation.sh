#!/bin/bash
# Evaluate Mono-T5 Reranker-Targeted Full RAG generation results

set -e

echo "Evaluating Mono-T5 Reranker-Targeted Full RAG generation results..."
echo ""

# Navigate to project root
cd "$(dirname "$0")/../../../../"

# Use project virtual environment
PYTHON=".venv/bin/python"

# Load environment variables from .env (including HF_TOKEN, OPENAI_API_KEY)
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set HuggingFace cache to scripts/generation_scripts directory
export HF_HOME="$(pwd)/scripts/baselines/generation_scripts/.cache/huggingface"
export HF_DATASETS_CACHE="$(pwd)/scripts/baselines/generation_scripts/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$(pwd)/scripts/baselines/generation_scripts/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=0  # Disable hf-xet to avoid crashes

RESULTS_DIR="scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag/results"

# Evaluate all non-evaluated result files
for result_file in ${RESULTS_DIR}/*.jsonl; do
    # Skip already evaluated files
    if [[ ! "$result_file" =~ _evaluated\.jsonl$ ]]; then
        base_name=$(basename "$result_file" .jsonl)
        output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
        
        # Skip if already evaluated
        if [ -f "$output_file" ]; then
            echo "Already evaluated: $base_name"
            continue
        fi
        
        echo "Evaluating: $base_name"
        
        $PYTHON scripts/evaluation/run_generation_eval.py \
            -i "$result_file" \
            -o "$output_file" \
            -e scripts/evaluation/config.yaml \
            --provider openai
        
        echo ""
    fi
done

echo "All evaluations complete!"
echo ""
echo "Evaluated results:"
ls -lh ${RESULTS_DIR}/*_evaluated.jsonl 2>/dev/null || echo "No evaluated files found yet."
