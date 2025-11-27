# MMR Clustering vs Query Rewrite Analysis

| Method              | Strategy                  | R@1   | R@3   | R@5   | R@10  | nDGC@1 | nDGC@3 | nDGC@5 | nDGC@10 |
|:--------------------|:--------------------------|:------|:------|:------|:------|:-------|:-------|:-------|:--------|
| BM25 (PyTerrier)    | Query Rewrite             | 0.090 | 0.189 | 0.250 | 0.349 | 0.211  | 0.200  | 0.223  | 0.264   |
|                     | MMR Cluster (Rewrite k5)  | 0.072 | 0.165 | 0.219 | 0.298 | 0.165  | 0.171  | 0.192  | 0.225   |
|                     | MMR Cluster (Rewrite k10) | 0.077 | 0.175 | 0.232 | 0.320 | 0.179  | 0.182  | 0.205  | 0.242   |
|                     | MMR Cluster (Rewrite k15) | 0.087 | 0.180 | 0.239 | 0.311 | 0.200  | 0.189  | 0.213  | 0.243   |
|                     | MMR Cluster (Rewrite k20) | 0.082 | 0.180 | 0.234 | 0.317 | 0.188  | 0.187  | 0.208  | 0.244   |
| BGE-base 1.5 (Ours) | Query Rewrite             | 0.153 | 0.305 | 0.381 | 0.498 | 0.358  | 0.326  | 0.354  | 0.404   |
|                     | MMR Cluster (Rewrite k5)  | 0.130 | 0.270 | 0.342 | 0.448 | 0.300  | 0.284  | 0.313  | 0.358   |
|                     | MMR Cluster (Rewrite k10) | 0.140 | 0.282 | 0.357 | 0.464 | 0.320  | 0.297  | 0.326  | 0.371   |
|                     | MMR Cluster (Rewrite k15) | 0.140 | 0.277 | 0.355 | 0.460 | 0.326  | 0.297  | 0.326  | 0.372   |
|                     | MMR Cluster (Rewrite k20) | 0.142 | 0.284 | 0.354 | 0.459 | 0.326  | 0.303  | 0.329  | 0.373   |
| Elser (Ours)        | Query Rewrite             | 0.187 | 0.372 | 0.476 | 0.608 | 0.429  | 0.399  | 0.438  | 0.495   |
|                     | MMR Cluster (Rewrite k5)  | 0.171 | 0.347 | 0.448 | 0.559 | 0.384  | 0.368  | 0.407  | 0.455   |
|                     | MMR Cluster (Rewrite k10) | 0.178 | 0.358 | 0.464 | 0.577 | 0.407  | 0.381  | 0.422  | 0.470   |
|                     | MMR Cluster (Rewrite k15) | 0.166 | 0.352 | 0.455 | 0.577 | 0.388  | 0.373  | 0.412  | 0.464   |
|                     | MMR Cluster (Rewrite k20) | 0.166 | 0.353 | 0.451 | 0.566 | 0.383  | 0.373  | 0.409  | 0.460   |

## Analysis

* **Consistent Underperformance:** The MMR Cluster strategy consistently underperforms the standard Query Rewrite
  baseline across all retrieval methods (Sparse, Dense, and Learned Sparse) and almost all metrics.
* **Noise Introduction:** Applying MMR clustering likely filters out relevant information captured
  by the standard Query Rewrite, leading to lower precision and recall.
* **Method-Specific Trends:**
    * **BM25:** Performance stagnates with increased `k`.
    * **BGE-base 1.5:** Slight improvement with higher `k`, but remains significantly below baseline (~8% drop in nDCG).
    * **Elser:** Performance peaks at `k10` and degrades with higher `k`, suggesting too much diversity hurts this
      learned sparse retriever.

##  Other Experiments
- Adjust the Diversity Parameter (Lambda)
  - Current Issue: The scripts default to --lambda-param 0.7 (70% relevance, 30% diversity).
  - Experiment: Run a sweep of lambda values (e.g., 0.3, 0.5, 0.85, 0.9).
  Hypothesis: For query rewriting, "relevance" to the current query is often more important than "diversity" of history. A higher lambda (e.g., 0.85 or 0.9) might prevent the inclusion of irrelevant historical tangents that distract the LLM.
- "Context Summary" vs. "Query Rewrite"
  - Current Issue: The current approach tries to rewrite the query into a standalone question.
  Experiment: Instead of rewriting the query, use the LLM to summarize the selected sentences into a "Context Block" and append it to the original query.
  - Format: Context: [Summary of selected history] \n Question: [Original User Query]
  Why? Rewriting can sometimes hallucinate or alter the user's intent. Providing context alongside the original query allows the retrieval model (especially dense ones) to match both the specific question and the context terms.
