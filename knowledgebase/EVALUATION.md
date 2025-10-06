# Evaluation Overview

## Setup

```bash
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```

## Retrieval Evaluation

**Script**: `scripts/evaluation/run_retrieval_eval.py`

**Metrics**: Recall@k, nDCG@k (k=1,3,5,10)

**Input**: JSONL file with `contexts` field containing:
- `document_id`: Passage identifier
- `score`: Retrieval score

**Output**: Per-collection metrics + aggregate CSV

**Usage**:
```bash
python scripts/evaluation/run_retrieval_eval.py \
  --input_file human/generation_tasks/RAG.jsonl \
  --output_file results.jsonl
```

## Generation Evaluation

**Script**: `scripts/evaluation/run_generation_eval.py`

**Metrics**: LLM-as-Judge evaluation (correctness, relevance, groundedness)

**Input**: JSONL file with `predictions` field containing:
- `text`: Generated response

**Usage with OpenAI**:
```bash
python scripts/evaluation/run_generation_eval.py \
  -i input.jsonl -o output.jsonl \
  -e scripts/evaluation/config.yaml \
  --provider openai --openai_key KEY --azure_host ENDPOINT
```

**Usage with HuggingFace**:
```bash
python scripts/evaluation/run_generation_eval.py \
  -i input.jsonl -o output.jsonl \
  -e scripts/evaluation/config.yaml \
  --provider hf --judge_model ibm-granite/granite-3.3-8b-instruct
```

## Sample Files

- **Retrieval input**: `human/generation_tasks/RAG.jsonl`
- **Generation input**: `scripts/evaluation/responses-10.jsonl` (10 samples)
- **Results**: `human/evaluations/*.json` (paper results)
