# Multi-Strategy Retrieval Fusion (RRF)

Combines results from multiple query strategies using Reciprocal Rank Fusion to improve retrieval quality.

## Overview

Instead of selecting a single "best" query strategy, this approach fuses results from:
- **Last Turn**: Last user message only
- **Rewrite**: Rewritten standalone query

RRF (Reciprocal Rank Fusion) combines rankings without requiring comparable scores, making it ideal for fusing results across different strategies.

## Quick Start

### Run 2-Way Fusion (lastturn + rewrite)

```bash
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/run_bge_2way.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/run_bm25_2way.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/run_elser_2way.sh
```

### Run 3-Way Fusion (lastturn + rewrite + questions)

```bash
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/run_bge_fusion.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/run_bm25_fusion.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/run_elser_fusion.sh
```

### Evaluate Fusion Results

**2-Way Fusion:**
```bash
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/evaluate_bge_2way.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/evaluate_bm25_2way.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/evaluate_elser_2way.sh
```

**3-Way Fusion:**
```bash
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/evaluate_bge_fusion.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/evaluate_bm25_fusion.sh
bash scripts/ideas/retrieval_tasks/multi_strategy_fusion/evaluate_elser_fusion.sh
```

## Manual Usage

### 2-way Fusion (lastturn + rewrite)

```bash
python scripts/ideas/retrieval_tasks/multi_strategy_fusion/rrf_fusion.py \
  --input_files \
    scripts/baselines/retrieval_scripts/elser/results/elser_clapnq_lastturn.jsonl \
    scripts/baselines/retrieval_scripts/elser/results/elser_clapnq_rewrite.jsonl \
  --output_file scripts/ideas/retrieval_tasks/multi_strategy_fusion/2way/datasets/elser_clapnq_fusion_2way.jsonl \
  --collection_name "mt-rag-clapnq-elser-512-100-20240503" \
  --top_k 10 \
  --rrf_k 60
```

### 3-way Fusion (lastturn + rewrite + questions)

```bash
python scripts/ideas/retrieval_tasks/multi_strategy_fusion/rrf_fusion.py \
  --input_files \
    scripts/baselines/retrieval_scripts/elser/results/elser_clapnq_lastturn.jsonl \
    scripts/baselines/retrieval_scripts/elser/results/elser_clapnq_rewrite.jsonl \
    scripts/baselines/retrieval_scripts/elser/results/elser_clapnq_questions.jsonl \
  --output_file scripts/ideas/retrieval_tasks/multi_strategy_fusion/3way/datasets/elser_clapnq_fusion_3way.jsonl \
  --collection_name "mt-rag-clapnq-elser-512-100-20240503" \
  --top_k 10 \
  --rrf_k 60
```

## How RRF Works

### Formula

```
RRF_score(doc) = Σ 1/(k + rank_i)
                 i=1..N

where:
- rank_i = position of document in strategy i (1-indexed)
- k = constant (60, from Cormack et al. SIGIR 2009)
- N = number of strategies (2 in our case)
```

### Example

```
Strategy 1 (lastturn): [doc_A(rank=1), doc_B(rank=2), doc_C(rank=5)]
Strategy 2 (rewrite):  [doc_B(rank=1), doc_C(rank=3), doc_A(rank=8)]

RRF scores (k=60):
doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← Best (high in both)
doc_A: 1/(60+1) + 1/(60+8) = 0.0164 + 0.0147 = 0.0311  ← Good (high in one)
doc_C: 1/(60+5) + 1/(60+3) = 0.0154 + 0.0159 = 0.0313  ← Good (moderate in both)

Final ranking: [doc_B, doc_C, doc_A]
```

**Key insight:** Documents appearing in both strategies rank higher, even if not #1 in either.

## Input/Output Format

Both input and output use the standard MT-RAG retrieval format:

```json
{
  "task_id": "query_id",
  "Collection": "collection_name",
  "contexts": [
    {
      "document_id": "doc_id",
      "score": 0.0325,
      "text": "document text",
      "title": "document title",
      "source": "url"
    },
    ...
  ]
}
```

**Note:** Output `score` field contains RRF scores (not original retrieval scores).

## Directory Structure

```
multi_strategy_fusion/
├── rrf_fusion.py                        # Core fusion implementation
├── README.md                            # This file
├── FUSION_RESULTS_COMPARISON.md         # Complete results analysis
├── run_*_2way.sh                        # Run 2-way fusion scripts
├── run_*_fusion.sh                      # Run 3-way fusion scripts
├── evaluate_*_2way.sh                   # Evaluate 2-way fusion
├── evaluate_*_fusion.sh                 # Evaluate 3-way fusion
├── 2way/
│   ├── datasets/                        # 2-way fusion datasets (lastturn + rewrite)
│   │   ├── bge_*_fusion_2way.jsonl      # 12 files (3 retrievers × 4 domains)
│   │   ├── bm25_*_fusion_2way.jsonl
│   │   └── elser_*_fusion_2way.jsonl
│   └── results/                         # Evaluation outputs
│       ├── *_evaluated.jsonl            # Per-query scores
│       └── *_evaluated_aggregate.csv    # Aggregate metrics
└── 3way/
    ├── datasets/                        # 3-way fusion datasets (lastturn + rewrite + questions)
    │   ├── bge_*_fusion_3way.jsonl      # 12 files (3 retrievers × 4 domains)
    │   ├── bm25_*_fusion_3way.jsonl
    │   └── elser_*_fusion_3way.jsonl
    └── results/                         # Evaluation outputs
        ├── *_evaluated.jsonl
        └── *_evaluated_aggregate.csv
```

## Expected Performance

Based on MQRF-RAG (Yang et al. 2025) and RAG Fusion literature:

**Baseline (Rewrite - current best single strategy):**
- ELSER: R@5 = 0.48, nDCG@5 = 0.44
- BGE: R@5 = 0.38, nDCG@5 = 0.40
- BM25: R@5 = 0.26, nDCG@5 = 0.27

**Expected with Fusion:**
- +3-8% Recall@5 improvement
- +2-5% nDCG@5 improvement
- Better coverage across different query types

## Dependencies

None! This script only uses Python standard library (json, argparse, collections).

Works with existing retrieval results from:
- `scripts/baselines/retrieval_scripts/elser/results/`
- `scripts/baselines/retrieval_scripts/bge/results/`
- `scripts/baselines/retrieval_scripts/bm25/results/`

## References

- **Cormack et al.** "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.
- **Rackauckas.** "RAG Fusion: Better RAG with reciprocal reranking." 2023.
- **Yang et al.** "MQRF-RAG: Multi-Strategy Query Rewriting Framework for RAG." 2025.
- **Elasticsearch Hybrid Search:** Uses RRF for combining BM25 + kNN results.

## Implementation Details

See `/Users/pastil/Dev/Github/mt-rag-benchmark/knowledgebase/ideas/retrieval_tasks/multi_strategy_fusion.md` for detailed design document.

