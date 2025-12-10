# Mono-T5 Oracle Selection

## Overview

This directory contains an implementation of **mono-T5-based oracle selection** for choosing the best retriever (BM25, BGE, or ELSER) for each query.

## Approach

Instead of using ground truth qrels to determine which retriever performs best (the true oracle), we use **mono-T5** to predict relevance scores for each document-query pair, then select the retriever that appears to have the best results.

### Methodology

1. **Retrieve with all retrievers**: For each query, retrieve top-10 documents using BM25, BGE, and ELSER
2. **Score with mono-T5**: Use mono-T5 to score each document-query pair for relevance
3. **Calculate predicted recall@10**: For each retriever, calculate predicted recall@10 based on mono-T5 scores (treating score > 0.5 as relevant)
4. **Select best retriever**: Choose the retriever with the highest predicted recall@10
5. **Evaluate**: Compare mono-T5 selection performance to:
   - Individual retrievers (BM25, BGE, ELSER)
   - True oracle (ground truth selection)

## Files

- **`mono_t5_oracle_selection.py`** - Main script implementing the mono-T5 oracle selection
- **`utils.py`** - Utility functions for data loading, metrics calculation, and JSON export
- **`download_model.py`** - Script to download and cache the mono-T5 model locally
- **`.cache/`** - Directory for cached model files (created by download_model.py)
- **`results/`** - Output directory for analysis results (JSON files)
  - `monot5_choices.json` - Task-level retriever selections by mono-T5
  - `predicted_recalls.json` - Predicted recall@10 for each retriever per task
  - `performance_metrics.json` - Performance metrics for individual retrievers, mono-T5 selection, and oracle
  - `selection_distributions.json` - Distribution of retriever selections (mono-T5 vs oracle)
  - `summary.json` - Summary statistics (improvements, gaps, etc.)

## Usage

### Step 1: Download and cache the model (first time only)

```bash
cd scripts/ideas/retrieval_tasks/mono_t5_oracle_selection
source ../../../../.venv/bin/activate  # or your virtual environment
python download_model.py
```

This will download the mono-T5 model and cache it in `.cache/` directory for faster subsequent runs.

### Step 2: Run the analysis

```bash
python mono_t5_oracle_selection.py
```

### Requirements

- PyTorch
- transformers (for mono-T5 model)
- sentencepiece (for T5 tokenizer)
- tqdm (for progress bars)

The script will:
1. Load retrieval results from all three retrievers (rewrite strategy)
2. Load queries from `human/retrieval_tasks/`
3. Initialize mono-T5 model from cache (or download if not cached)
4. Score all document-query pairs
5. Select best retriever for each query
6. Save results as JSON files

## Expected Output

The script generates JSON files in the `results/` directory:

### JSON Output Files

1. **`monot5_choices.json`** - Maps each task_id to the retriever selected by mono-T5
   ```json
   {
     "task_id_1": "bm25",
     "task_id_2": "bge",
     ...
   }
   ```

2. **`predicted_recalls.json`** - Predicted recall@10 for each retriever per task
   ```json
   {
     "task_id_1": {
       "bm25": 0.8,
       "bge": 0.6,
       "elser": 0.7
     },
     ...
   }
   ```

3. **`performance_metrics.json`** - Aggregated performance metrics
   ```json
   {
     "individual_retrievers": {
       "bm25": {"recall_10": 0.65, ...},
       "bge": {"recall_10": 0.70, ...},
       "elser": {"recall_10": 0.68, ...}
     },
     "monot5_selection": {"recall_10": 0.72, ...},
     "oracle": {"recall_10": 0.75, ...}
   }
   ```

4. **`selection_distributions.json`** - Distribution of retriever selections
   ```json
   {
     "monot5_selection": {
       "bm25": {"count": 100, "percentage": 25.0},
       ...
     },
     "oracle_selection": {...}
   }
   ```

5. **`summary.json`** - Key summary statistics
   ```json
   {
     "monot5_recall_10": 0.72,
     "best_individual_recall_10": 0.70,
     "improvement_over_best_percent": 2.86,
     "oracle_recall_10": 0.75,
     "gap_to_oracle": 0.03,
     "gap_to_oracle_percent": 4.0,
     "total_tasks": 400,
     "strategy": "rewrite"
   }
   ```

## Research Questions

This implementation helps answer:

1. **Can mono-T5 effectively predict retriever quality?** - Does mono-T5 selection outperform individual retrievers?

2. **How close can we get to oracle?** - What's the gap between mono-T5 selection and true oracle performance?

3. **Is retriever selection learnable?** - If mono-T5 can approximate oracle selection, it suggests that learned retriever selection is feasible

## Notes

- Currently uses **rewrite strategy** only (can be extended to other strategies)
- Uses **recall@10** as the selection metric (can be changed to nDCG@5 or other metrics)
- Mono-T5 scoring treats documents with score > 0.5 as "relevant" (threshold can be tuned)
- Processing all queries can take significant time depending on GPU availability

## Related Work

- See `knowledgebase/retrieval/mono-t5-reranker/` for oracle analysis using ground truth
- Mono-T5 model: [castorini/monot5-base-msmarco](https://huggingface.co/castorini/monot5-base-msmarco)
