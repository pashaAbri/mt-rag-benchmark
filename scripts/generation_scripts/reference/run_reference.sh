#!/bin/bash
# Run generation on Reference setting (perfect retriever - 842 tasks)
# Uses gold reference passages only, no retrieval

set -e

# Navigate to project root
cd "$(dirname "$0")/../../.."

MODEL_CONFIG="scripts/generation_scripts/model_invocation/llm_configs/gpt4o.yaml"
INPUT_FILE="human/generation_tasks/reference.jsonl"
OUTPUT_FILE="scripts/generation_scripts/reference/results/gpt_4o_reference.jsonl"

echo "=========================================="
echo "Reference Setting - Perfect Retriever"
echo "=========================================="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model Config: $MODEL_CONFIG"
echo ""

python scripts/generation_scripts/model_invocation/llm_caller.py \
    --model_config "$MODEL_CONFIG" \
    --prompt_file "scripts/generation_scripts/model_invocation/prompts/baseline.py" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 10 \
    --concurrency 5 \
    --resume

echo ""
echo "✓ Reference generation complete!"
echo "Results saved to: $OUTPUT_FILE"

