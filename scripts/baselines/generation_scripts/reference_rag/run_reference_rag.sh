#!/bin/bash
# Run generation on Reference+RAG setting (hybrid - 436 tasks)
# Uses gold reference passages + retrieved passages

set -e

# Navigate to project root
cd "$(dirname "$0")/../../../../"

MODEL_CONFIG="scripts/baselines/generation_scripts/model_invocation/llm_configs/command_r_plus.yaml"
INPUT_FILE="human/generation_tasks/reference+RAG.jsonl"
OUTPUT_FILE="scripts/baselines/generation_scripts/reference_rag/results/command_r_plus_reference_rag.jsonl"

echo "=========================================="
echo "Reference+RAG Setting - Hybrid Retrieval"
echo "=========================================="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model Config: $MODEL_CONFIG"
echo "Note: Subset of 436 tasks with ≤2 reference passages"
echo ""

python scripts/baselines/generation_scripts/model_invocation/llm_caller.py \
    --model_config "$MODEL_CONFIG" \
    --prompt_file "scripts/baselines/generation_scripts/model_invocation/prompts/baseline.py" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 10 \
    --concurrency 5 \
    --resume

echo ""
echo "✓ Reference+RAG generation complete!"
echo "Results saved to: $OUTPUT_FILE"

