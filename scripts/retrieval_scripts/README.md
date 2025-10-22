# Retrieval Scripts

This directory contains implementations for various retrieval baselines on the MT-RAG benchmark.

## Structure

```
scripts/retrieval_scripts/
├── bm25/                        # BM25 lexical retrieval (PyTerrier) ✅ COMPLETE
│   ├── bm25_retrieval.py        # Main BM25 implementation
│   ├── run_bm25_all.sh          # Run all BM25 experiments (4 domains × 3 query types)
│   ├── evaluate_bm25.sh         # Evaluate all BM25 results
│   ├── results/                 # BM25 outputs (12 experiments + evaluations)
│   └── README.md                # BM25-specific documentation
│
├── bge/                         # BGE-base 1.5 dense retrieval (TODO)
│   ├── bge_retrieval.py         # Main BGE implementation
│   ├── results/                 # BGE outputs (when ready)
│   └── README.md                # BGE-specific documentation
│
├── elser/                       # Elser sparse retrieval (TODO)
│   ├── elser_retrieval.py       # Main Elser implementation
│   ├── results/                 # Elser outputs (when ready)
│   └── README.md                # Elser-specific documentation
│
├── utils.py                     # Shared utilities for all retrievers
└── run_all_baselines.py         # Master script to run all retrievers
```

## Quick Start

### BM25 (Available Now)

Run all BM25 experiments:
```bash
bash scripts/retrieval_scripts/bm25/run_bm25_all.sh
bash scripts/retrieval_scripts/bm25/evaluate_bm25.sh
```

### Other Retrievers (Coming Soon)

BGE and Elser implementations are placeholders and will be organized in similar subdirectories.

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

