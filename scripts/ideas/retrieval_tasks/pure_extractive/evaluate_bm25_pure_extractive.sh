#!/bin/bash
# Evaluate BM25 retrieval results for pure extractive queries

set -e

# Navigate to workspace root
cd "$(dirname "$0")"/../../../..

echo "======================================"
echo "Evaluating BM25 Pure Extractive Results"
echo "======================================"
echo ""

for result_file in scripts/ideas/retrieval_tasks/pure_extractive/results/bm25_*.jsonl; do
    # Skip already evaluated files
    if [[ ! "$result_file" =~ _evaluated\.jsonl$ ]]; then
        base_name=$(basename "$result_file" .jsonl)
        output_file="scripts/ideas/retrieval_tasks/pure_extractive/results/${base_name}_evaluated.jsonl"
        
        echo "Evaluating: $base_name"
        python scripts/evaluation/run_retrieval_eval.py \
            --input_file "$result_file" \
            --output_file "$output_file"
        echo ""
    fi
done

echo "======================================"
echo "Evaluation Complete!"
echo "======================================"
echo ""
echo "Aggregate results:"
cat scripts/ideas/retrieval_tasks/pure_extractive/results/bm25_*_evaluated_aggregate.csv
echo ""

