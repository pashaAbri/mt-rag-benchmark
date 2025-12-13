# Targeted Query Rewrite

## Hypothesis

The baseline query rewrite indiscriminately includes the **entire conversation history** when rewriting queries. This can introduce noise and irrelevant context that hurts retrieval.

**Targeted Rewrite** selectively includes only the previous turns that are **semantically relevant** to the current query. By filtering out unrelated context, we hypothesize that:

1. The rewritten query will be more focused and less noisy
2. Retrieval performance will improve, especially for long conversations
3. Queries that shift topic mid-conversation will benefit most

## Method

### Pipeline Overview

```
Conversation History          Current Query
        │                          │
        ▼                          ▼
┌───────────────────────────────────────────┐
│  Step 1: Embed with Sentence Transformer  │
│         (all-MiniLM-L6-v2)                │
└───────────────────────────────────────────┘
        │                          │
        ▼                          ▼
   Turn Embeddings           Query Embedding
        │                          │
        └──────────┬───────────────┘
                   ▼
┌───────────────────────────────────────────┐
│  Step 2: Compute Cosine Similarity        │
│  between query and each history turn      │
└───────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│  Step 3: Filter Relevant Turns            │
│  • Keep turns with similarity ≥ threshold │
│  • Always include last turn (recency)     │
│  • Cap at max_relevant_turns              │
└───────────────────────────────────────────┘
                   │
                   ▼
         Filtered Context
                   │
                   ▼
┌───────────────────────────────────────────┐
│  Step 4: LLM Query Rewriting              │
│  Pass only relevant context to Claude     │
└───────────────────────────────────────────┘
                   │
                   ▼
        Standalone Rewritten Query
```

### Step 1: Load and Structure Conversation History

For each task (query turn), we load all previous user and agent messages from the conversation. The history is grouped into **turn pairs**, where each pair consists of a user message and the corresponding agent response:

```
Turn 0: User: "Tell me about the Arizona Cardinals"
        Agent: "The Arizona Cardinals are an NFL team based in..."

Turn 1: User: "When were they founded?"
        Agent: "The Cardinals were founded in 1898..."

Turn 2: User: "Do they play outside the US?"  ← Current query
```

### Step 2: Compute Semantic Similarity

We use a lightweight sentence transformer (`all-MiniLM-L6-v2`) to embed both the current query and each previous turn pair. For turn pairs, we concatenate the user and agent messages:

```
Turn text = "User: {user_message} Assistant: {agent_response}"
```

We then compute **cosine similarity** between the current query embedding and each turn embedding:

```python
similarity = dot(query_embedding, turn_embedding) / (||query|| * ||turn||)
```

This produces a similarity score in the range [-1, 1] for each historical turn, indicating how semantically related that turn is to the current query.

### Step 3: Filter Relevant Turns

The filtering logic applies three rules:

1. **Similarity Threshold**: Include turns where `similarity ≥ threshold` (default: 0.3). This filters out turns that discuss unrelated topics.

2. **Recency Bias**: Always include the immediately preceding turn, regardless of similarity. The last turn often contains important context even if semantically distant.

3. **Maximum Turns Cap**: Limit to at most `max_relevant_turns` (default: 5) to prevent context overload for the LLM.

The selected turns are returned in **chronological order** to maintain narrative coherence.

**Example**: In a 6-turn conversation about multiple topics:

- Turns 0-1: Discussion about NFL history
- Turns 2-3: Discussion about stadium locations (similarity: 0.45, 0.52)
- Turn 4: Discussion about team mascots (similarity: 0.18)
- Current query: "Which stadiums have retractable roofs?"

Selected turns: [2, 3, 4] — Turns 2-3 are semantically relevant; Turn 4 is included due to recency.

### Step 4: LLM Query Rewriting

The filtered relevant turns are passed to Claude with a prompt instructing it to rewrite the current query into a standalone form:

```
System: You are an expert at rewriting conversational queries into standalone queries.

Given a conversation history and the user's current question, rewrite the current
question into a standalone query that:
1. Contains all necessary context from the conversation to be understood independently
2. Is clear and self-contained
3. Preserves the user's original intent
4. Does NOT introduce new information not present in the conversation

If the query is already standalone and doesn't need context, return it unchanged.

Output ONLY the rewritten query, nothing else.
```

By providing only the relevant context (rather than the full history), the LLM produces more focused rewrites without being distracted by unrelated earlier discussion.

### Step 5: Retrieval and Evaluation

The rewritten queries are run through multiple retrieval systems:

- **BM25** (sparse lexical matching via PyTerrier)
- **BGE** (dense embedding retrieval)
- **ELSER** (Elasticsearch learned sparse retrieval)

Performance is measured using standard IR metrics (nDCG@k, Recall@k) and compared against the baseline rewrite approach.

## Configuration Options

| Parameter                | Default                      | Description                                   |
| ------------------------ | ---------------------------- | --------------------------------------------- |
| `--similarity_threshold` | 0.3                          | Minimum cosine similarity to include a turn   |
| `--max_relevant_turns`   | 5                            | Maximum number of relevant turns to include   |
| `--include_last_turn`    | True                         | Always include the immediately preceding turn |
| `--embedding_model`      | `all-MiniLM-L6-v2`           | Sentence transformer model for embeddings     |
| `--llm_model`            | `claude-sonnet-4-5-20250929` | LLM for query rewriting                       |

## Expected Results

We expect the largest gains for:

- **Long conversations** (5+ turns) where topic drift is likely
- **Topic-switching queries** where the current question is unrelated to earlier turns
- **Govt/Cloud domains** which have more technical, focused queries

Minimal change expected for:

- **Turn 1 queries** (no history to filter)
- **Short conversations** (2-3 turns) where all context is typically relevant
