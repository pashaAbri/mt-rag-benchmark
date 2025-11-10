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

## Results

**📊 [MASTER_RESULTS_SUMMARY.md](MASTER_RESULTS_SUMMARY.md) - Complete analysis across BM25 and BGE**

### Quick Performance Summary

#### BM25 (Lexical) - NDCG@10

| Domain | Lastturn | Human | Pure | Hybrid | Best |
|--------|----------|-------|------|--------|------|
| clapnq | 0.269 | **0.301** | 0.290 ↑ | 0.284 ↑ | Human |
| cloud | **0.252** | 0.248 | 0.239 ↓ | 0.241 ↓ | Lastturn |
| fiqa | 0.136 | **0.186** | 0.152 ↑ | 0.155 ↑ | Human |
| govt | 0.319 | **0.354** | 0.339 ↑ | 0.336 ↑ | Human |
| **Avg** | 0.244 | **0.272** | 0.255 | 0.254 | Human |

#### BGE (Semantic) - NDCG@10

| Domain | Lastturn | Human | Pure | Hybrid | Best |
|--------|----------|-------|------|--------|------|
| clapnq | 0.424 | **0.498** | 0.406 ↓ | 0.399 ↓ | Human |
| cloud | 0.307 | **0.342** | 0.285 ↓ | 0.290 ↓ | Human |
| fiqa | 0.291 | **0.341** | 0.234 ↓ | 0.236 ↓ | Human |
| govt | 0.344 | **0.420** | 0.306 ↓ | 0.303 ↓ | Human |
| **Avg** | 0.342 | **0.400** | 0.308 | 0.307 | Human |

### Key Findings:

**1. Pure ≈ Hybrid** (both retrievers)
- BM25: 0.4% difference
- BGE: 0.3% difference
- **Conclusion:** Templates/NER don't help

**2. Extractive better with BM25**
- BM25 gap from human: -6.3%  ✅ Competitive
- BGE gap from human: -23.0%  ❌ Not competitive

**3. BGE improves absolute scores**
- Pure: +21% average improvement
- Hybrid: +21% average improvement
- But widens gap from baselines

**4. Domain matters**
- Best: govt (BM25), clapnq (BGE)
- Worst: cloud (both), fiqa (BGE)

### Final Recommendation:

✅ **Use Pure Extractive + BM25** for production (simple, competitive)  
❌ **Avoid Hybrid** (no benefit over Pure)  
❌ **Avoid Extractive + BGE** (too far behind baselines)

### Detailed Analysis:
- [BM25 Results](FULL_RESULTS_COMPARISON.md)
- [BGE Results](BGE_RESULTS_COMPARISON.md)
- [Master Summary](MASTER_RESULTS_SUMMARY.md)

## Documentation

See `knowledgebase/ideas/retrieval_tasks/` for detailed documentation:
- `extractive_implementation_analysis.md` - Data analysis and approach rationale
- `pure_extractive_rewrite.md` - Pure extractive implementation details
- `hybrid_extractive_rewrite.md` - Hybrid extractive implementation details

