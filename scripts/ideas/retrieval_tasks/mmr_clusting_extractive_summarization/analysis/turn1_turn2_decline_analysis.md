# Analysis: Sharp Decline Between Turn 1 and Turn 2

## Overview

Found **52 examples** across all domains where Turn 1 had high recall (≥0.5) but Turn 2 showed a decline of ≥0.10. The most extreme cases show a complete drop from 1.0 → 0.0 recall.

## Key Patterns Identified

### 1. **Topic Shift Problem** (Most Common)

**Example: North Carolina Weather**
- **Turn 1**: "where does it snow the most in north carolina?" → Recall: 1.0
- **Turn 2**: "What can you tell me about the summer months. Are they enjoyable?" → Recall: 0.0
- **Problem**: The selected sentences from Turn 1 are all about **snow** (the previous topic), but Turn 2 is asking about **summer months** (completely different topic).
- **Selected Sentences**: All 4 sentences mention snow, mountains, precipitation - nothing about summer.

**Example: MLS All-Star Game**
- **Turn 1**: "how does the mls all star game work" → Recall: 1.0  
- **Turn 2**: "What is it called outside of the USA?" → Recall: 0.0
- **Problem**: Turn 2 shifts from asking about how the game works to asking about naming conventions in other countries - a topic shift that may not be well-covered in the corpus.

### 2. **Vague Follow-up Queries**

**Example: The Office**
- **Turn 1**: "what happens to toby at the end of the office" → Recall: 1.0
- **Turn 2**: "any reason to end?" → Recall: 0.0
- **Problem**: Turn 2 query is extremely vague. The rewriting doesn't help: "Any reason to end?" - unclear what "end" refers to (the show? the character arc?).

### 3. **Context Pollution**

The MMR clustering is selecting sentences from Turn 1 that are relevant to the **previous** query, not the current one. This creates "context pollution" where:

- The rewritten query incorporates irrelevant context from Turn 1
- The retrieval system searches for documents matching the rewritten query + irrelevant context
- Results are poor because the corpus doesn't contain documents matching this mixed signal

### 4. **Rewriting Quality Issues**

**Example: Watson Assistant**
- **Turn 1**: Elaborate query about Watson Assistant → Recall: 1.0
- **Turn 2**: "Does he understand my emotions?" → Rewritten: "Does the assistant have the ability to understand my emotions?"
- **Problem**: The rewriting is reasonable, but the selected context from Turn 1 is about API connectivity and technical setup, not emotional understanding capabilities.

## Statistics

- **CLAPNQ**: 17 examples (33%)
- **FIQA**: 16 examples (31%)  
- **CLOUD**: 11 examples (21%)
- **GOVT**: 8 examples (15%)

## Root Causes

1. **No Topic Change Detection**: The system doesn't detect when Turn 2 is asking about a completely different topic than Turn 1.

2. **MMR Selection Bias**: MMR is selecting sentences that are relevant to the **previous** query (Turn 1) rather than the **current** query (Turn 2).

3. **Limited Context Window**: With only 1 turn of history, the system has very little context to work with, making it prone to over-fitting to the previous topic.

4. **Query Rewriting Assumptions**: The LLM rewriting assumes that all selected sentences are relevant to the current query, but they may not be.

## Recommendations

1. **Topic Change Detection**: Add a mechanism to detect when Turn 2 is asking about a different topic than Turn 1. In such cases, use minimal or no context from Turn 1.

2. **Query-Relevance Filtering**: Before applying MMR, filter sentences by their relevance to the **current** query, not just their diversity.

3. **Turn 2 Special Handling**: Consider treating Turn 2 differently - it's the first turn where context is introduced, and the system may need more conservative context selection.

4. **Rewriting Validation**: After rewriting, validate that the rewritten query makes sense given the selected context. If there's a mismatch, reduce context or re-select.

## Example Cases for Further Investigation

See `turn1_turn2_decline_examples.json` for all 52 examples with full details including:
- Original queries
- Rewritten queries  
- Selected sentences
- Context statistics

