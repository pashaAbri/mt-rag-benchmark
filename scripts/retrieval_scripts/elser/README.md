# Elser Sparse Retrieval

Elser (ElasticSearch Learned Sparse EncodeR) retrieval implementation for the MT-RAG benchmark.

## Model

- **Name:** ELSERv1
- **Type:** Learned sparse retrieval
- **Platform:** ElasticSearch 8.10+
- **Documentation:** https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html

## Status

🚧 **TODO:** Implementation in progress

## Requirements

- ElasticSearch 8.10 or higher
- ELSERv1 model deployed in ElasticSearch
- Python elasticsearch client

## Planned Usage

```bash
# Run single experiment
python scripts/retrieval_scripts/elser/elser_retrieval.py \
    --domain clapnq \
    --query_type lastturn \
    --query_file human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl \
    --output_file scripts/retrieval_scripts/elser/results/elser_clapnq_lastturn.jsonl \
    --es_host http://localhost:9200 \
    --index_name mt-rag-clapnq

# Run all experiments
bash scripts/retrieval_scripts/elser/run_elser_all.sh

# Evaluate results
bash scripts/retrieval_scripts/elser/evaluate_elser.sh
```

## Implementation Notes

- Requires ElasticSearch server running locally or remotely
- Corpora must be indexed with Elser model (512 token chunks, 100 token overlap)
- Uses text_expansion query type for Elser retrieval

