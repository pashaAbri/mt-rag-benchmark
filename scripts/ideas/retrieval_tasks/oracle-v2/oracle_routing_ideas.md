# Oracle Routing Ideas

## Motivation

Based on our post-retrieval oracle analysis, we know that intelligent routing significantly improves retrieval performance:
- **Single-retriever oracle** (selecting best strategy per query): +17.22% improvement on nDCG@10
- **Cross-retriever oracle** (selecting best retriever+strategy combination): +30.42% improvement on nDCG@10

However, the oracle analysis requires:
1. Running all retrievals first (computationally expensive)
2. Access to ground truth relevance judgments (qrels) to determine what's "good"

## Goal

Design a **pre-retrieval routing system** that can predict the optimal retriever+strategy combination for each query **before** performing any retrievals, using only query characteristics and available metadata.

## Available Features for Routing

### Query Characteristics
- Query length (word count, character count)
- Turn number in conversation
- Query content (keywords, entities, semantic features)
- Query structure (question type indicators, punctuation patterns)

### Conversation Context Metadata
- **Enrichments** (see `knowledgebase/enrichments.md` for details):
  - **Answerability**: ANSWERABLE, UNANSWERABLE, PARTIAL, CONVERSATIONAL
  - **Question Type**: Factoid, Explanation, Composite, Comparative, How-To, Keyword, Opinion, Summarization, Troubleshooting, Non-Question
  - **Multi-Turn Type**: Follow-up, Clarification, N/A (first turn)
- Conversation history length
- Previous turn characteristics
- Domain (clapnq, cloud, fiqa, govt)

### Pattern Analysis
- Analyze oracle selection patterns to identify correlations between query features and optimal combinations
- Examine cases where BM25 outperforms ELSER (28.3% of queries)
- Investigate when Lastturn vs Rewrite is optimal (59.5% vs 25.9%)
- Study domain-specific patterns and query type preferences

## Next Steps

1. **Feature Engineering**: Extract and normalize features from query and conversation metadata
2. **Pattern Discovery**: Analyze oracle selections to identify predictive patterns
3. **Model Development**: Train classifiers/routers using query features to predict optimal combinations
4. **Evaluation**: Measure how well the routing system captures oracle performance gains
