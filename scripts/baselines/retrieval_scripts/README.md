# Retrieval Scripts

This directory contains implementations for various retrieval baselines on the MT-RAG benchmark.

## Structure

```
scripts/baselines/retrieval_scripts/
├── bm25/                        # BM25 lexical retrieval (PyTerrier)
│   ├── bm25_retrieval.py        # Main BM25 implementation
│   ├── run_bm25_all.sh          # Run all BM25 experiments (4 domains × 3 query types)
│   ├── evaluate_bm25.sh         # Evaluate all BM25 results
│   ├── results/                 # BM25 outputs (12 experiments + evaluations)
│   └── README.md                # BM25-specific documentation
│
├── bge/                         # BGE-base 1.5 dense retrieval
│   ├── bge_retrieval.py         # Main BGE implementation
│   ├── run_bge_all.sh           # Run all BGE experiments
│   ├── evaluate_bge.sh          # Evaluate all BGE results
│   ├── results/                 # BGE outputs (12 experiments + evaluations)
│   └── README.md                # BGE-specific documentation
│
├── elser/                       # ELSER learned sparse retrieval
│   ├── elser_retrieval.py       # Main ELSER implementation
│   ├── run_elser_all.sh         # Run all ELSER experiments (4 domains)
│   ├── evaluate_elser.sh        # Evaluate all ELSER results
│   ├── results/                 # ELSER outputs (12 experiments + evaluations)
│   └── README.md                # ELSER-specific documentation
│
├── utils.py                     # Shared utilities for all retrievers
└── run_all_baselines.py         # Master script to run all retrievers
```

## Quick Start

### BM25 (Available Now)

Run all BM25 experiments:
```bash
bash scripts/baselines/retrieval_scripts/bm25/run_bm25_all.sh
bash scripts/baselines/retrieval_scripts/bm25/evaluate_bm25.sh
```

### BGE (Dense Retrieval)

Run all BGE experiments:
```bash
bash scripts/baselines/retrieval_scripts/bge/run_bge_all.sh
bash scripts/baselines/retrieval_scripts/bge/evaluate_bge.sh
```

### ELSER (Learned Sparse Retrieval)

Run all ELSER experiments (all 4 domains):
```bash
bash scripts/baselines/retrieval_scripts/elser/run_elser_all.sh
bash scripts/baselines/retrieval_scripts/elser/evaluate_elser.sh
```

**Note:** ELSER requires Elasticsearch Cloud with `.env` credentials configured.
**Results:** R@5=0.44, nDCG@5=0.41 (closely matches paper's baseline of 0.49/0.45)

## Output Format

All scripts output results in the format expected by `scripts/evaluation/run_retrieval_eval.py`:

```json
{
  "task_id": "...",
  "Collection": "...",
  "contexts": [
    {
      "document_id": "...",
      "score": float,
      "text": "...",
      "title": "..."
    }
  ]
}
```

