# Pure Extractive Query Rewriting

MMR-based term selection for conversational query rewriting.

## Quick Start

### 1. Run BM25 Retrieval

```bash
cd scripts/ideas/retrieval_tasks/pure_extractive
./run_bm25_pure_extractive.sh
```

### 2. Evaluate Results

```bash
cd scripts/ideas/retrieval_tasks/pure_extractive
./evaluate_bm25_pure_extractive.sh
```

### Step-by-Step Guide

#### 1. Generate Datasets Only

```bash
cd scripts/ideas/retrieval_tasks
./generate_all_extractive_datasets.sh
```

Outputs: `pure_extractive/datasets/{domain}_pure_extractive.jsonl`

#### 2. Run BM25 Retrieval

```bash
cd scripts/ideas/retrieval_tasks/pure_extractive
./run_bm25_pure_extractive.sh
```

Outputs: `results/bm25_{domain}_pure_extractive.jsonl`

#### 3. Evaluate Results

```bash
cd scripts/ideas/retrieval_tasks/pure_extractive
./evaluate_bm25_pure_extractive.sh
```

Outputs:
- `results/bm25_{domain}_pure_extractive_evaluated.jsonl`
- `results/bm25_{domain}_pure_extractive_evaluated_aggregate.csv`

## Method

**Pure Extractive** uses:
- MMR (Maximal Marginal Relevance) for term selection
- Extracts unigrams, bigrams, and trigrams from query + conversation history
- Balances relevance (λ=0.7) vs diversity
- Selects top 10 terms
- Concatenates terms into keyword query

**Example:**
- **Original:** "Do the Arizona Cardinals play outside the US?"
- **History:** "where do the arizona cardinals play this week"
- **Rewritten:** "where arizona cardinals play outside us cardinals play outside arizona..."

## Output Structure

```
pure_extractive/
├── pure_extractive_rewrite.py          # Main implementation
├── datasets/                            # Generated query rewrites
│   ├── clapnq_pure_extractive.jsonl    
│   ├── cloud_pure_extractive.jsonl
│   ├── fiqa_pure_extractive.jsonl
│   └── govt_pure_extractive.jsonl
├── results/                             # BM25 retrieval outputs
│   ├── bm25_clapnq_pure_extractive.jsonl
│   ├── bm25_clapnq_pure_extractive_evaluated.jsonl
│   ├── bm25_clapnq_pure_extractive_evaluated_aggregate.csv
│   └── ... (same for other domains)
├── run_bm25_pure_extractive.sh         # Run BM25 retrieval
└── evaluate_bm25_pure_extractive.sh    # Evaluate results
```

## Dependencies

- sentence-transformers (BAAI/bge-base-en-v1.5)
- nltk (stopwords only)
- scikit-learn
- numpy

Install: `pip install sentence-transformers nltk scikit-learn numpy`

## Comparison with Baselines

To compare with baseline retrieval methods:
- **Lastturn:** `human/retrieval_tasks/{domain}/{domain}_lastturn.jsonl`
- **Human rewrite:** `human/retrieval_tasks/{domain}/{domain}_rewrite.jsonl`
- **Pure extractive:** `datasets/{domain}_pure_extractive.jsonl`

