#!/bin/bash
# Evaluate BGE 2-way fusion results

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

echo "=========================================="
echo "Evaluating BGE 2-Way Fusion Results"
echo "=========================================="
echo ""

for result_file in scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/datasets/bge_*_fusion_2way.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/results/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python scripts/evaluation/run_retrieval_eval.py \
        --input_file "$result_file" \
        --output_file "$output_file"
    echo ""
done

echo "=========================================="
echo "✓ Evaluation Complete!"
echo "=========================================="
echo ""
echo "Aggregate results:"
cat scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/results/bge_*_evaluated_aggregate.csv
echo ""

