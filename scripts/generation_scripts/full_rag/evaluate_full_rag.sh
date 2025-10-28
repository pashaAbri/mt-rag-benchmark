#!/bin/bash
# Evaluate Full RAG generation results

set -e

echo "Evaluating Full RAG generation results..."
echo ""

# Navigate to project root
cd "$(dirname "$0")/../../.."

for result_file in scripts/generation_scripts/full_rag/results/llama_3.1_8b_*.jsonl; do
    # Skip already evaluated files
    if [[ ! "$result_file" =~ _evaluated\.jsonl$ ]]; then
        base_name=$(basename "$result_file" .jsonl)
        output_file="scripts/generation_scripts/full_rag/results/${base_name}_evaluated.jsonl"
        
        echo "Evaluating: $base_name"
        
        python scripts/evaluation/run_generation_eval.py \
            -i "$result_file" \
            -o "$output_file" \
            -e scripts/evaluation/config.yaml \
            --provider hf \
            --judge_model "ibm-granite/granite-3.3-8b-instruct"
        
        echo ""
    fi
done

echo "All evaluations complete!"
echo ""
echo "Evaluated results:"
ls -lh scripts/generation_scripts/full_rag/results/llama_3.1_8b_*_evaluated.jsonl

