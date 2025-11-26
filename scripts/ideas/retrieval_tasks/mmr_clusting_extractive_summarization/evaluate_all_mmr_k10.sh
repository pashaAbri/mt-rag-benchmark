#!/bin/bash
# Evaluate all retrieval methods (BM25, BGE, ELSER) for MMR-rewritten queries (k=10)

# Ensure .venv is activated if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

set -e

RESULTS_DIR="scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results_k10"
SCRIPT_DIR="scripts/evaluation"

echo "=========================================="
echo "Evaluating ALL retrieval methods (k=10)"
echo "=========================================="
echo ""

# Remove any existing evaluation files first to ensure a clean run
# Only do this once at the beginning
echo "Cleaning up existing evaluation files..."
rm -f "${RESULTS_DIR}"/*_evaluated.jsonl
rm -f "${RESULTS_DIR}"/*_evaluated_aggregate.csv
echo ""

# ==========================================
# BM25 Evaluation
# ==========================================
echo "=========================================="
echo "1. Evaluating BM25 results..."
echo "=========================================="

# Combine BM25 results first
echo "Combining BM25 results..."
chmod +x scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_bm25_mmr_results_k10.sh
./scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_bm25_mmr_results_k10.sh
echo ""

# Evaluate BM25 results
for result_file in "${RESULTS_DIR}"/bm25_*_mmr_cluster_k10.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
done
echo "✓ BM25 evaluation complete!"
echo ""

# ==========================================
# BGE Evaluation
# ==========================================
echo "=========================================="
echo "2. Evaluating BGE results..."
echo "=========================================="

# Combine BGE results first
echo "Combining BGE results..."
chmod +x scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_bge_mmr_results_k10.sh
./scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_bge_mmr_results_k10.sh
echo ""

# Evaluate BGE results
for result_file in "${RESULTS_DIR}"/bge_*_mmr_cluster_k10.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
done
echo "✓ BGE evaluation complete!"
echo ""

# ==========================================
# ELSER Evaluation
# ==========================================
echo "=========================================="
echo "3. Evaluating ELSER results..."
echo "=========================================="

# Combine ELSER results first
echo "Combining ELSER results..."
chmod +x scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_elser_mmr_results_k10.sh
./scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/combine_elser_mmr_results_k10.sh
echo ""

# Evaluate ELSER results
for result_file in "${RESULTS_DIR}"/elser_*_mmr_cluster_k10.jsonl; do
    base_name=$(basename "$result_file" .jsonl)
    output_file="${RESULTS_DIR}/${base_name}_evaluated.jsonl"
    
    echo "Evaluating: $base_name"
    python "${SCRIPT_DIR}/run_retrieval_eval.py" \
        --input_file "$result_file" \
        --output_file "$output_file"
done
echo "✓ ELSER evaluation complete!"
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""
echo "Summary of evaluated files:"
echo "  BM25: $(ls "${RESULTS_DIR}"/bm25_*_evaluated.jsonl 2>/dev/null | wc -l | tr -d ' ') files"
echo "  BGE:  $(ls "${RESULTS_DIR}"/bge_*_evaluated.jsonl 2>/dev/null | wc -l | tr -d ' ') files"
echo "  ELSER: $(ls "${RESULTS_DIR}"/elser_*_evaluated.jsonl 2>/dev/null | wc -l | tr -d ' ') files"
echo ""
echo "Aggregate results (head of each CSV):"
head -n 2 "${RESULTS_DIR}"/*_evaluated_aggregate.csv 2>/dev/null || echo "No aggregate files found"

