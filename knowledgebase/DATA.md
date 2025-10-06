# Data Overview

## Corpora (4 Domains)

| Domain | Description | Documents | Passages |
|--------|-------------|-----------|----------|
| **ClapNQ** | Wikipedia | 4,293 | 183,408 |
| **Cloud** | Technical Docs | 57,638 | 61,022 |
| **FiQA** | Finance | 7,661 | 49,607 |
| **Govt** | Government | 8,578 | 72,422 |

Location: `corpora/passage_level/*.jsonl.zip`

## Human-Generated Data

**110 conversations** → **842 evaluation tasks**

### Conversations
- Location: `human/conversations/conversations.json`
- Average: 7.7 turns per conversation
- Single domain per conversation

### Retrieval Tasks
- Location: `human/retrieval_tasks/[domain]/`
- BEIR format: queries + qrels
- Three query variants: rewrite, last turn, all questions

### Generation Tasks
- Location: `human/generation_tasks/`
- Files: `reference.jsonl`, `reference+RAG.jsonl`, `RAG.jsonl`
- Each task = one conversation turn with full history

## Synthetic Data

- **200 conversations**: `synthetic/conversations/`
- Generation tasks: `synthetic/generation_tasks/synthetic.jsonl`

## Question Properties

- **Answerability**: Answerable, Unanswerable, Partial, Conversational
- **Question Type**: Factoid, Explanation, Composite, Keyword, Opinion, etc.
- **Multi-Turn**: Follow-up, Clarification
