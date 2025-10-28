#!/bin/bash
# Evaluate Reference generation results

set -e

echo "Evaluating Reference generation results..."
echo ""

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Load environment variables from .env (including HF_TOKEN)
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set HuggingFace cache to scripts/generation_scripts directory
export HF_HOME="$(pwd)/scripts/generation_scripts/.cache/huggingface"
export HF_DATASETS_CACHE="$(pwd)/scripts/generation_scripts/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$(pwd)/scripts/generation_scripts/.cache/huggingface/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=0  # Disable hf-xet to avoid crashes

for result_file in scripts/generation_scripts/reference/results/llama_3.1_8b_*.jsonl; do
    # Skip already evaluated files
    if [[ ! "$result_file" =~ _evaluated\.jsonl$ ]]; then
        base_name=$(basename "$result_file" .jsonl)
        output_file="scripts/generation_scripts/reference/results/${base_name}_evaluated.jsonl"
        
        echo "Evaluating: $base_name"
        
        python scripts/evaluation/run_generation_eval_v2.py \
            -i "$result_file" \
            -o "$output_file" \
            -e scripts/evaluation/config.yaml \
            --provider hf \
            --judge_model "together_ai"
        
        echo ""
    fi
done

echo "All evaluations complete!"
echo ""
echo "Evaluated results:"
ls -lh scripts/generation_scripts/reference/results/llama_3.1_8b_*_evaluated.jsonl

