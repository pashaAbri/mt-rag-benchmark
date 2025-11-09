#!/bin/bash
# Evaluate all BM25 retrieval results

set -e

echo "Evaluating BM25 retrieval results..."
echo ""

for result_file in scripts/baselines/retrieval_scripts/bm25/results/bm25_*.jsonl; do
    if [[ ! "$result_file" =~ _evaluated\.jsonl$ ]]; then
        base_name=$(basename "$result_file" .jsonl)
        output_file="scripts/baselines/retrieval_scripts/bm25/results/${base_name}_evaluated.jsonl"
        
        echo "Evaluating: $base_name"
        python scripts/evaluation/run_retrieval_eval.py \
            --input_file "$result_file" \
            --output_file "$output_file"
        echo ""
    fi
done

echo "All evaluations complete!"
echo ""
echo "Aggregate results:"
cat scripts/baselines/retrieval_scripts/bm25/results/bm25_*_evaluated_aggregate.csv

