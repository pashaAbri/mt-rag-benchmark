#!/bin/bash
# Run generation on Full RAG setting (real-world - 842 tasks)
# Uses pure retrieval (top 5 passages from Elser)

set -e

# Navigate to project root
cd "$(dirname "$0")/../../.."

MODEL_CONFIG="scripts/generation_scripts/model_invocation/llm_configs/llama_3.1_405b.yaml"
INPUT_FILE="human/generation_tasks/RAG.jsonl"
OUTPUT_FILE="scripts/generation_scripts/full_rag/results/llama_3.1_405b_full_rag.jsonl"

echo "=========================================="
echo "Full RAG Setting - Real-World Pipeline"
echo "=========================================="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model Config: $MODEL_CONFIG"
echo "Note: End-to-end RAG with Elser retrieval"
echo ""

python scripts/generation_scripts/model_invocation/llm_caller.py \
    --model_config "$MODEL_CONFIG" \
    --prompt_file "scripts/generation_scripts/model_invocation/prompts/baseline.py" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 10 \
    --resume

echo ""
echo "✓ Full RAG generation complete!"
echo "Results saved to: $OUTPUT_FILE"

