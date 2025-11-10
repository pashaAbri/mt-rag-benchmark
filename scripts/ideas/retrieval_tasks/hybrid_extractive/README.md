# Hybrid Extractive Query Rewriting

MMR-based term selection + Templates + Named Entity Recognition for conversational query rewriting.

## Quick Start

### 1. Generate Datasets

```bash
cd scripts/ideas/retrieval_tasks/hybrid_extractive
python hybrid_extractive_rewrite.py clapnq
python hybrid_extractive_rewrite.py cloud
python hybrid_extractive_rewrite.py fiqa
python hybrid_extractive_rewrite.py govt
```

Or generate all at once:
```bash
cd scripts/ideas/retrieval_tasks
./generate_all_extractive_datasets.sh
```
(Note: Current version only generates pure extractive; update script to include hybrid)

### 2. Run BM25 Retrieval

```bash
cd scripts/ideas/retrieval_tasks/hybrid_extractive
./run_bm25_hybrid_extractive.sh
```

Outputs: `results/bm25_{domain}_hybrid_extractive.jsonl`

### 3. Evaluate Results

```bash
cd scripts/ideas/retrieval_tasks/hybrid_extractive
./evaluate_bm25_hybrid_extractive.sh
```

Outputs:
- `results/bm25_{domain}_hybrid_extractive_evaluated.jsonl`
- `results/bm25_{domain}_hybrid_extractive_evaluated_aggregate.csv`

## Method

**Hybrid Extractive** extends Pure Extractive with:
- **Named Entity Recognition (NER)** - Identifies people, places, organizations using spaCy
- **Entity Boosting** - Increases relevance score for entities by 1.5x
- **Question Type Classification** - Detects what/where/when/who/why/how questions
- **Template-based Reformulation** - Applies appropriate templates based on question type
- **Post-processing** - Capitalizes, adds question marks, deduplicates words

**Example:**
- **Original:** "Do the Arizona Cardinals play outside the US?"
- **History:** "where do the arizona cardinals play this week"
- **Entities Detected:** "Arizona Cardinals" (ORG), "US" (GPE)
- **Question Type:** "do" question
- **Rewritten:** "Do Arizona Cardinals play outside US?"

## Output Structure

```
hybrid_extractive/
├── hybrid_extractive_rewrite.py        # Main implementation
├── datasets/                            # Generated query rewrites
│   ├── clapnq_hybrid_extractive.jsonl    
│   ├── cloud_hybrid_extractive.jsonl
│   ├── fiqa_hybrid_extractive.jsonl
│   └── govt_hybrid_extractive.jsonl
├── results/                             # BM25 retrieval outputs
│   ├── bm25_clapnq_hybrid_extractive.jsonl
│   ├── bm25_clapnq_hybrid_extractive_evaluated.jsonl
│   ├── bm25_clapnq_hybrid_extractive_evaluated_aggregate.csv
│   └── ... (same for other domains)
├── run_bm25_hybrid_extractive.sh       # Run BM25 retrieval
└── evaluate_bm25_hybrid_extractive.sh  # Evaluate results
```

## Dependencies

- sentence-transformers (BAAI/bge-base-en-v1.5)
- spacy (en_core_web_sm) - for NER
- nltk (stopwords only)
- scikit-learn
- numpy

Install: 
```bash
pip install sentence-transformers spacy nltk scikit-learn numpy
python -m spacy download en_core_web_sm
```

## Comparison with Pure Extractive

| Feature | Pure Extractive | Hybrid Extractive |
|---------|----------------|-------------------|
| N-gram extraction | ✅ | ✅ |
| MMR selection | ✅ | ✅ |
| Named Entity Recognition | ❌ | ✅ |
| Entity boosting | ❌ | ✅ (1.5x) |
| Question templates | ❌ | ✅ |
| Well-formed output | ❌ (keywords) | ✅ (questions) |

## Baseline Comparison

To compare with baseline retrieval methods:
- **Lastturn:** `human/retrieval_tasks/{domain}/{domain}_lastturn.jsonl`
- **Human rewrite:** `human/retrieval_tasks/{domain}/{domain}_rewrite.jsonl`
- **Pure extractive:** `../pure_extractive/datasets/{domain}_pure_extractive.jsonl`
- **Hybrid extractive:** `datasets/{domain}_hybrid_extractive.jsonl`

