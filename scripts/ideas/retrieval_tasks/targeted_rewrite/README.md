# Targeted Query Rewrite

## Hypothesis

The baseline query rewrite indiscriminately includes the **entire conversation history** when rewriting queries. This can introduce noise and irrelevant context that hurts retrieval.

**Targeted Rewrite** selectively includes only the previous turns that are **semantically relevant** to the current query. By filtering out unrelated context, we hypothesize that:

1. The rewritten query will be more focused and less noisy
2. Retrieval performance will improve, especially for long conversations
3. Queries that shift topic mid-conversation will benefit most

## Method

1. **Load Conversation History**
   - For each task (query turn), load all previous user and agent messages

2. **Compute Semantic Similarity**
   - Embed the current query using a sentence transformer (e.g., `all-MiniLM-L6-v2`)
   - Embed each previous turn (user + agent pair)
   - Compute cosine similarity between current query and each previous turn

3. **Filter Relevant Turns**
   - Include a previous turn if similarity > threshold (default: 0.3)
   - Optionally limit to top-K most relevant turns
   - Always include the immediately preceding turn (recency bias option)

4. **Rewrite Query**
   - Pass only the filtered relevant turns to the LLM for rewriting
   - Use the same rewrite prompt as baseline

5. **Evaluate**
   - Generate retrieval results using the targeted rewrite queries
   - Compare against baseline rewrite using nDCG@k and Recall@k

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--similarity_threshold` | 0.3 | Minimum cosine similarity to include a turn |
| `--max_relevant_turns` | 5 | Maximum number of relevant turns to include |
| `--include_last_turn` | True | Always include the immediately preceding turn |
| `--embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `--llm_model` | `claude-sonnet-4-5-20250929` | LLM for query rewriting |

## Usage

### Generate targeted rewrites:
```bash
python scripts/ideas/retrieval_tasks/targeted_rewrite/run_targeted_rewrite.py \
    --domains clapnq cloud fiqa govt \
    --similarity_threshold 0.3 \
    --max_relevant_turns 5
```

### Run with different thresholds:
```bash
# More selective (fewer turns)
python run_targeted_rewrite.py --similarity_threshold 0.5

# Less selective (more turns)  
python run_targeted_rewrite.py --similarity_threshold 0.2
```

### Evaluate results:
```bash
python scripts/evaluation/run_retrieval_eval.py \
    --input_file intermediate/targeted_rewrite_clapnq.jsonl \
    --output_file intermediate/targeted_rewrite_clapnq_evaluated.jsonl
```

## Output

Results are saved in `intermediate/`:

- `targeted_rewrite_{domain}.jsonl`: Rewritten queries in BEIR format
- `targeted_rewrite_{domain}_analysis.json`: Detailed analysis with per-query metadata

### BEIR Query Format (`targeted_rewrite_{domain}.jsonl`)
```json
{"_id": "task_id", "text": "|user|: rewritten query"}
```

### Analysis Format (`targeted_rewrite_{domain}_analysis.json`)
```json
{
  "config": {
    "similarity_threshold": 0.3,
    "max_relevant_turns": 5,
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "claude-sonnet-4-5-20250929"
  },
  "stats": {
    "turn1_no_rewrite": 29,
    "rewritten": 195,
    "turns_filtered": 120
  },
  "analyses": [
    {
      "task_id": "...",
      "turn_id": 2,
      "original_query": "Do they play outside the US?",
      "rewritten_query": "Do the Arizona Cardinals play outside the United States?",
      "method": "targeted_rewrite",
      "num_history_turns": 3,
      "selected_turns": 2,
      "selected_indices": [0, 2],
      "similarities": [...]
    }
  ]
}

## Expected Results

We expect the largest gains for:
- **Long conversations** (5+ turns) where topic drift is likely
- **Topic-switching queries** where the current question is unrelated to earlier turns
- **Govt/Cloud domains** which have more technical, focused queries

Minimal change expected for:
- **Turn 1 queries** (no history to filter)
- **Short conversations** (2-3 turns) where all context is typically relevant

