# Tasks Overview

## What is the Task?

**Multi-Turn Conversational RAG**: Build systems that can maintain context across conversation turns while retrieving and generating accurate responses.

## Task Types

### Retrieval Task
- Given a conversation history, retrieve the most relevant passages from a corpus
- Input: Multi-turn conversation (queries)
- Output: Ranked list of document passages
- Format: BEIR-compatible

### Generation Task
- Given conversation history + retrieved passages, generate an accurate response
- Input: Conversation history + contexts (passages)
- Output: Natural language response
- Three settings:
  - **Reference**: Use gold-standard passages
  - **Reference + RAG**: Mix reference + retrieved passages
  - **Full RAG**: Use only retrieved passages

## Key Characteristics

Each task contains:
- Full conversation history up to current turn
- Retrieved/reference passages (contexts)
- Human-annotated reference answer
- Question metadata: answerability, type, multi-turn dimension

## Challenge

Handle **follow-up questions** ("How many?", "Who is their coach?") and **clarifications** that require understanding previous conversation context.
