#!/bin/bash
# Evaluate all ELSER retrieval results

set -e

echo "Evaluating ELSER retrieval results..."
echo ""

for result_file in scripts/retrieval_scripts/elser/results/elser_*.jsonl; do
    if [[ ! "$result_file" =~ _evaluated\.jsonl$ ]]; then
        base_name=$(basename "$result_file" .jsonl)
        output_file="scripts/retrieval_scripts/elser/results/${base_name}_evaluated.jsonl"
        
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
cat scripts/retrieval_scripts/elser/results/elser_*_evaluated_aggregate.csv

