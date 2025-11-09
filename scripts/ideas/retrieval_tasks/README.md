# Extractive Query Rewriting - Scripts

This directory contains implementations of extractive query rewriting approaches for the MT-RAG benchmark.

## Directory Structure

- `pure_extractive/` - MMR-based term selection
- `hybrid_extractive/` - MMR + Templates + Entity Recognition
- `generate_all_extractive_datasets.sh` - Generate all datasets at once

## Quick Start

### 1. Download the Embedding Model (First Time Only)

```bash
cd scripts/ideas/retrieval_tasks
python download_model.py
```

This downloads the BGE model (~400MB) to `.models/bge-base-en-v1.5` for offline use.

### 2. Generate All Datasets

To generate datasets for all domains and both methods:

```bash
cd scripts/ideas/retrieval_tasks
./generate_all_extractive_datasets.sh
```

This will:
1. Check/download the model if needed
2. Generate rewrites for all domains
3. Create:
   - Analysis results in `{method}/results/`
   - MT-RAG formatted datasets in `{method}/datasets/`

### Individual Method Usage

#### Pure Extractive

```bash
python pure_extractive/pure_extractive_rewrite.py <domain>
```

Example:
```bash
python pure_extractive/pure_extractive_rewrite.py clapnq
```

Generates rewrites using MMR-based term selection.

**Outputs:**
- Analysis: `pure_extractive/results/{domain}_rewrites.jsonl`
- MT-RAG format: `pure_extractive/datasets/{domain}_pure_extractive.jsonl`

#### Hybrid Extractive

```bash
python hybrid_extractive/hybrid_extractive_rewrite.py <domain>
```

Example:
```bash
python hybrid_extractive/hybrid_extractive_rewrite.py cloud
```

Generates rewrites using MMR + templates + NER.

**Outputs:**
- Analysis: `hybrid_extractive/results/{domain}_rewrites.jsonl`
- MT-RAG format: `hybrid_extractive/datasets/{domain}_hybrid_extractive.jsonl`

## Domains

- `clapnq` - Wikipedia Q&A (208 queries)
- `cloud` - Technical Documentation (188 queries)
- `fiqa` - Financial Q&A (180 queries)
- `govt` - Government/Policy (201 queries)

## Dependencies

```bash
pip install sentence-transformers spacy scikit-learn nltk
python -m spacy download en_core_web_sm
```

**Note**: Both methods use `BAAI/bge-base-en-v1.5` embeddings, which will be downloaded automatically on first run (~400MB).

## Output Directory Structure

After running the generation scripts, you'll have:

```
scripts/ideas/retrieval_tasks/
├── pure_extractive/
│   ├── pure_extractive_rewrite.py
│   ├── datasets/                         (MT-RAG format for retrieval)
│   │   ├── clapnq_pure_extractive.jsonl (208 queries)
│   │   ├── cloud_pure_extractive.jsonl  (188 queries)
│   │   ├── fiqa_pure_extractive.jsonl   (180 queries)
│   │   └── govt_pure_extractive.jsonl   (201 queries)
│   └── results/                          (Analysis format)
│       ├── clapnq_rewrites.jsonl
│       ├── cloud_rewrites.jsonl
│       ├── fiqa_rewrites.jsonl
│       └── govt_rewrites.jsonl
│
└── hybrid_extractive/
    ├── hybrid_extractive_rewrite.py
    ├── datasets/                         (MT-RAG format for retrieval)
    │   ├── clapnq_hybrid_extractive.jsonl
    │   ├── cloud_hybrid_extractive.jsonl
    │   ├── fiqa_hybrid_extractive.jsonl
    │   └── govt_hybrid_extractive.jsonl
    └── results/                          (Analysis format)
        ├── clapnq_rewrites.jsonl
        ├── cloud_rewrites.jsonl
        ├── fiqa_rewrites.jsonl
        └── govt_rewrites.jsonl
```

## Output Formats

### Analysis Format (results/)

Detailed analysis with original and rewritten queries:

```json
{
  "id": "query_id",
  "original": "original query text",
  "rewritten": "rewritten query text",
  "history_length": 3
}
```

### MT-RAG Format (datasets/)

Standard format for retrieval evaluation:

```json
{
  "_id": "query_id",
  "text": "|user|: rewritten query text"
}
```

This format matches the existing MT-RAG retrieval tasks and can be used directly with retrieval systems.

## Documentation

See `knowledgebase/ideas/retrieval_tasks/` for detailed documentation:
- `extractive_implementation_analysis.md` - Data analysis and approach rationale
- `pure_extractive_rewrite.md` - Pure extractive implementation details
- `hybrid_extractive_rewrite.md` - Hybrid extractive implementation details

