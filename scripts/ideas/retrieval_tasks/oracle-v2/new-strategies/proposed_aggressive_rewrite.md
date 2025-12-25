# Proposal: Aggressive Retrieval-Optimized Rewrite Strategy

## Motivation
Analysis of 98 zero-score retrieval cases revealed that the current rewrite strategy fails because it prioritizes "minimalism" and "naturalness." 
- 47% of rewrites added no context.
- 15% failed to resolve pronouns.

## Proposed Prompt
This prompt is designed to force **entity resolution** and **context injection**, favoring retrieval performance over conversational flow.

```text
You are a Search Engine Query Optimizer. Your task is to rewrite the last user utterance into a fully standalone search query that will retrieve relevant documents.

Rules:
1. **RESOLVE ALL PRONOUNS**: Replace every pronoun (it, he, she, they, this, that) with the specific entity name it refers to from the conversation history.
2. **INJECT MISSING CONTEXT**: If the query implies a topic discussed earlier (e.g., "what about security?"), explicitly add the topic (e.g., "IBM Cloud security features").
3. **SPECIFY GENERIC TERMS**: Replace generic words like "the series", "the act", "the company" with their full proper names (e.g., "The Office US", "The Affordable Care Act").
4. **IGNORE NATURALNESS**: The output does not need to sound like a natural conversation. It must be an effective keyword-rich search query.
5. **NEVER OUTPUT THE SAME QUERY**: If the user's query relies on *any* previous context, you MUST modify it.

Input Conversation:
{{conversation}}

Output ONLY the rewritten query text.
```

## Expected Improvements
| Failure Case | Original | Baseline Rewrite | Aggressive Rewrite (Projected) |
| :--- | :--- | :--- | :--- |
| **Unresolved Pronoun** | "Why did it fall?" | "Why did it fall?" | "Why did the **Byzantine Empire** fall?" |
| **Generic Term** | "any awards" | "Did the series win any awards?" | "Did **The Office US TV series** win any awards?" |
| **No Context** | "What about security?" | "What about security?" | "security features of **IBM Cloud**" |

