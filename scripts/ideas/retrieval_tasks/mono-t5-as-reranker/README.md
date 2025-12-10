# Combined Retrieval + MonoT5 Reranking

Evaluates a strategy that combines multiple retrieval queries and reranks them with MonoT5.

## Hypothesis
Combining diverse retrieval strategies (`lastturn`, `rewrite`, `questions`) increases recall. Reranking with MonoT5 (using the `rewrite` query) filters noise and improves ranking quality.

## Method
1.  **Combine**: Merge retrieval results from all three strategies for each task.
2.  **Deduplicate**: Remove duplicate documents.
3.  **Rerank**: Score all unique documents using `monot5-base-msmarco` against the `rewrite` query.
4.  **Evaluate**: Run standard benchmark evaluation on the top-100 reranked results.

## Usage

```bash
pip install torch transformers pytrec-eval tqdm
cd scripts/ideas/retrieval_tasks/mono-t5-as-reranker
python evaluate_rerank.py
```

## Output
Results are saved in `intermediate/`:
*   `reranked_{domain}.jsonl`: Raw reranked results.
*   `reranked_{domain}_evaluated.jsonl`: Results with metrics.
*   `reranked_{domain}_evaluated_aggregate.csv`: Aggregate metrics (nDCG, Recall).

## Key Findings
*   **Overall**: **+5.9% nDCG@10** improvement vs. best single strategy.
*   **Govt**: **+11.2%** improvement (fixes query drift).
*   **Cloud**: **+7.6%** improvement.
*   **ClapNQ**: **+1.1%** improvement (baseline already strong).

See `knowledgebase/retrieval/mono-t5-reranker/performance_chart.md` for detailed metrics.
