# Combined Retrieval + MonoT5 Reranking

Evaluates a strategy that combines multiple retrieval queries and reranks them with MonoT5.

## Hypothesis
Combining diverse retrieval strategies (`lastturn`, `rewrite`, `questions`) increases recall. Reranking with MonoT5 (using the `rewrite` query) filters noise and improves ranking quality.

## Method

1.  **Combine Retrieval Results**
    *   Loads retrieval results from three distinct query strategies for each domain:
        *   `lastturn`: Uses only the most recent user question.
        *   `rewrite`: Uses a manually or LLM-rewritten standalone query.
        *   `questions`: Concatenates all questions in the conversation history.
    *   Merges the lists of retrieved documents for each task ID into a single pool.

2.  **Deduplicate Documents**
    *   Uses `document_id` as the unique identifier.
    *   Removes duplicate entries to ensure each document appears only once in the candidate pool.
    *   Preserves the text content and metadata of the document.

3.  **Rerank with MonoT5**
    *   **Model**: Uses `castorini/monot5-base-msmarco`, a T5-base model fine-tuned on MS MARCO for passage ranking.
    *   **Query Selection**: Uses the `rewrite` query text as the reference query for scoring, as it provides the most complete standalone context.
    *   **Scoring**:
        *   Constructs input pairs: `Query: {query_text} Document: {doc_text} Relevant:`
        *   Computes the probability of the model generating the token "true" vs "false".
        *   Assigns the "true" probability as the relevance score.
    *   **Ranking**: Sorts all documents in the pool by descending relevance score.
    *   **Truncation**: Retains only the top 100 documents for the final output.

4.  **Evaluate Performance**
    *   **Standardization**: Formats the reranked results into the standard JSONL format required by the benchmark evaluation suite.
    *   **Execution**: Invokes `scripts/evaluation/run_retrieval_eval.py` to ensure consistent metric calculation.
    *   **Metrics**: Calculates nDCG@k and Recall@k (k=1,3,5,10) using `pytrec_eval` against the official domain qrels.

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
