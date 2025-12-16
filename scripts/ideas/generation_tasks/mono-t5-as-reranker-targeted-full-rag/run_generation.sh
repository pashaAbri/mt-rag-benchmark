#!/bin/bash
# Run generation on Mono-T5 Reranker-Targeted Full RAG setting
# Uses top-5 passages from mono-t5 reranker-targeted (3-strategy)

set -e

# Navigate to project root
cd "$(dirname "$0")/../../../../"

# Use project virtual environment
PYTHON=".venv/bin/python"

# Configuration
MODEL_CONFIG="scripts/baselines/generation_scripts/model_invocation/llm_configs/gpt4o.yaml"
INPUT_FILE="scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag/mono_t5_targeted_RAG.jsonl"
OUTPUT_FILE="scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag/results/gpt_4o_mono_t5_targeted_full_rag.jsonl"

echo "=========================================="
echo "Mono-T5 Reranker-Targeted Full RAG"
echo "=========================================="
echo ""

# Step 1: Prepare generation tasks (if not already done)
if [ ! -f "$INPUT_FILE" ]; then
    echo "Step 1: Preparing generation tasks..."
    $PYTHON scripts/ideas/generation_tasks/mono-t5-as-reranker-targeted-full-rag/prepare_generation_tasks.py \
        --output "$INPUT_FILE" \
        --top_k 5
    echo ""
else
    echo "Step 1: Generation tasks already prepared ($INPUT_FILE)"
    echo ""
fi

# Step 2: Run generation
echo "Step 2: Running generation with GPT-4o..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model Config: $MODEL_CONFIG"
echo ""

$PYTHON scripts/baselines/generation_scripts/model_invocation/llm_caller.py \
    --model_config "$MODEL_CONFIG" \
    --prompt_file "scripts/baselines/generation_scripts/model_invocation/prompts/baseline.py" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 10 \
    --concurrency 5 \
    --resume

echo ""
echo "✓ Generation complete!"
echo "Results saved to: $OUTPUT_FILE"
