# Extractive Query Rewriting - Scripts

This directory contains implementations of extractive query rewriting approaches for the MT-RAG benchmark.

## Directory Structure

- `pure_extractive/` - MMR-based term selection
- `hybrid_extractive/` - MMR + Templates + Entity Recognition

## Usage

### Pure Extractive

```bash
python pure_extractive/pure_extractive_rewrite.py <domain>
```

Example:
```bash
python pure_extractive/pure_extractive_rewrite.py clapnq
```

Generates rewrites using MMR-based term selection.

**Output:** `scripts/ideas/retrieval_tasks/pure_extractive/results/{domain}_rewrites.jsonl`

### Hybrid Extractive

```bash
python hybrid_extractive/hybrid_extractive_rewrite.py <domain>
```

Example:
```bash
python hybrid_extractive/hybrid_extractive_rewrite.py cloud
```

Generates rewrites using MMR + templates + NER.

**Output:** `scripts/ideas/retrieval_tasks/hybrid_extractive/results/{domain}_rewrites.jsonl`

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

## Results

Each experiment saves results in its own `results/` subdirectory:

```
scripts/ideas/retrieval_tasks/
├── pure_extractive/
│   ├── pure_extractive_rewrite.py
│   └── results/                          (generated)
│       ├── clapnq_rewrites.jsonl        (208 queries)
│       ├── cloud_rewrites.jsonl         (188 queries)
│       ├── fiqa_rewrites.jsonl          (180 queries)
│       └── govt_rewrites.jsonl          (201 queries)
│
└── hybrid_extractive/
    ├── hybrid_extractive_rewrite.py
    └── results/                          (generated)
        ├── clapnq_rewrites.jsonl        (208 queries)
        ├── cloud_rewrites.jsonl         (188 queries)
        ├── fiqa_rewrites.jsonl          (180 queries)
        └── govt_rewrites.jsonl          (201 queries)
```

### Output Format

Each rewrite file contains JSONL with:
```json
{
  "id": "query_id",
  "original": "original query text",
  "rewritten": "rewritten query text",
  "history_length": 3
}
```

## Documentation

See `knowledgebase/ideas/retrieval_tasks/` for detailed documentation:
- `extractive_implementation_analysis.md` - Data analysis and approach rationale
- `pure_extractive_rewrite.md` - Pure extractive implementation details
- `hybrid_extractive_rewrite.md` - Hybrid extractive implementation details

