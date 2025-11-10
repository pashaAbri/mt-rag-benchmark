#!/bin/bash
# Evaluate BM25 3-way fusion results

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

echo "=========================================="
echo "Evaluating BM25 3-Way Fusion Results"
echo "=========================================="
echo ""

for result_file in scripts/ideas/retrieval_tasks/multi_strategy_fusion/3way/datasets/bm25_*_fusion_3way.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="scripts/ideas/retrieval_tasks/multi_strategy_fusion/3way/results/${base_name}_evaluated.jsonl"
    
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
cat scripts/ideas/retrieval_tasks/multi_strategy_fusion/3way/results/bm25_*_evaluated_aggregate.csv
echo ""

