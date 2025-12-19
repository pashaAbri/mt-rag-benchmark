# SPLADE Retrieval Experiment

This directory contains an implementation of SPLADE (SParse Lexical AnD Expansion) retrieval for the MT-RAG benchmark.

## Overview

SPLADE is a learned sparse retrieval model that combines the efficiency and interpretability of sparse methods (like BM25) with the effectiveness of learned representations. It produces sparse term-level representations where each dimension corresponds to a vocabulary term.

**Key characteristics:**
- Learns to expand queries and documents with relevant terms
- Produces interpretable sparse representations  
- Uses dot product for efficient similarity computation
- Achieves state-of-the-art results on many IR benchmarks

**Default Model:** `naver/splade-cocondenser-ensembledistil` (~440 MB)

## Setup

### 1. Download the SPLADE model

```bash
cd /path/to/mt-rag-benchmark
source venv/bin/activate
python scripts/ideas/retrieval_tasks/splade/download_model.py
```

This will download the model to `scripts/ideas/retrieval_tasks/splade/models/`.

### 2. Run retrieval on all domains and query types

```bash
bash scripts/ideas/retrieval_tasks/splade/run_splade_all.sh
```

This runs SPLADE on:
- **4 domains:** clapnq, fiqa, govt, cloud
- **3 query types:** lastturn, rewrite, questions
- **Total:** 12 experiments

### 3. Evaluate results

```bash
bash scripts/ideas/retrieval_tasks/splade/evaluate_splade.sh
```

## File Structure

```
splade/
├── splade_retrieval.py       # Main SPLADE implementation
├── download_model.py         # Model download script
├── run_splade_all.sh        # Run all experiments
├── evaluate_splade.sh       # Evaluate all results
├── README.md                # This file
├── models/                  # Downloaded model (gitignored)
│   └── naver-splade-cocondenser-ensembledistil/
├── embeddings/              # Cached sparse embeddings (gitignored)
│   └── {domain}_splade_embeddings.npz
└── results/                 # Output retrieval results
    ├── splade_{domain}_{query_type}.jsonl
    ├── splade_{domain}_{query_type}_evaluated.jsonl
    └── splade_{domain}_{query_type}_evaluated_aggregate.csv
```

## Single Domain/Query Type

To run on a single domain and query type:

```bash
source venv/bin/activate
python scripts/ideas/retrieval_tasks/splade/splade_retrieval.py \
    --domain clapnq \
    --query_type lastturn \
    --corpus_file corpora/passage_level/clapnq.jsonl \
    --query_file human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl \
    --output_file scripts/ideas/retrieval_tasks/splade/results/splade_clapnq_lastturn.jsonl \
    --top_k 10
```

## Output Format

Results are saved in JSONL format compatible with the evaluation script:

```json
{
  "task_id": "query_id",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "doc_id",
      "score": 0.85,
      "text": "Document text...",
      "title": "Document title",
      "source": "URL or source"
    }
  ]
}
```

## Implementation Details

### SPLADE Encoding

The implementation follows the standard SPLADE encoding process:

1. Tokenize input text
2. Pass through a masked language model (MLM head)
3. Apply ReLU and log1p activation: `log(1 + ReLU(logits))`
4. Max-pool over sequence length (with attention mask)
5. Store as sparse representation

### Caching

- Corpus embeddings are cached in `embeddings/` directory
- Subsequent runs reuse cached embeddings (much faster)
- Delete cache to re-encode: `rm -rf scripts/ideas/retrieval_tasks/splade/embeddings/`

### Performance Considerations

- First run encodes entire corpus (can take 30+ minutes per domain)
- Query encoding is fast (~seconds for all queries)
- Sparse representations are memory-efficient
- Uses scipy sparse matrices for storage

## Comparison with Baselines

| Retriever | Type | Model Size |
|-----------|------|------------|
| BM25 | Lexical | N/A |
| BGE | Dense | ~420 MB |
| ELSER | Learned Sparse | Cloud-based |
| **SPLADE** | **Learned Sparse** | **~440 MB** |

## References

- [SPLADE Paper](https://arxiv.org/abs/2107.05720) - Formal et al., 2021
- [SPLADE v2](https://arxiv.org/abs/2109.10086) - Formal et al., 2021
- [HuggingFace Model](https://huggingface.co/naver/splade-cocondenser-ensembledistil)
