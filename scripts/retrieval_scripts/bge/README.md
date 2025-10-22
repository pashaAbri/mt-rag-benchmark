# BGE-base 1.5 Dense Retrieval

BGE (BAAI General Embedding) dense retrieval implementation for the MT-RAG benchmark.

## Model

- **Name:** BAAI/bge-base-en-v1.5
- **Type:** Dense retrieval using neural embeddings
- **Paper:** C-Pack: Packaged resources to advance general chinese embedding (Xiao et al., 2023)

## Status

🚧 **TODO:** Implementation in progress

## Planned Usage

```bash
# Run single experiment
python scripts/retrieval_scripts/bge/bge_retrieval.py \
    --domain clapnq \
    --query_type lastturn \
    --corpus_file corpora/passage_level/clapnq.jsonl \
    --query_file human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl \
    --output_file scripts/retrieval_scripts/bge/results/bge_clapnq_lastturn.jsonl

# Run all experiments
bash scripts/retrieval_scripts/bge/run_bge_all.sh

# Evaluate results
bash scripts/retrieval_scripts/bge/evaluate_bge.sh
```

## Implementation Notes

Will use:
- `sentence-transformers` library
- `BAAI/bge-base-en-v1.5` model from HuggingFace
- FAISS for efficient similarity search

