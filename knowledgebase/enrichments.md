# Enrichments in mtRAG

**Enrichments** are metadata annotations that human annotators added to each user question during the conversation creation process. They classify questions along three key dimensions that are central to the benchmark's design.

---

## Three Types of Enrichments

### 1. Answerability (4 categories)

Indicates whether the question can be answered given the retrieved passages.

| Category | Description | Example |
|----------|-------------|---------|
| **ANSWERABLE** | Question can be fully answered from passages | "Do the Arizona Cardinals play outside the US?" |
| **UNANSWERABLE** | Question cannot be answered from available passages | "where do the arizona cardinals play this week" (needs current info not in corpus) |
| **PARTIAL** | Question can be partially answered (some information available, but incomplete) | Questions where corpus has some but not all needed information |
| **CONVERSATIONAL** | Not a factual question; conversational/acknowledgment | "Thank you" or "I see" |

---

### 2. Question Type (10 categories)

Classifies the nature of the question.

| Type | Description |
|------|-------------|
| **Factoid** | Simple fact-based questions |
| **Explanation** | Requires detailed explanation |
| **Composite** | Multiple questions in one |
| **Comparative** | Comparing two or more things |
| **How-To** | Procedural/instructional questions |
| **Keyword** | Keyword-based search queries |
| **Opinion** | Subjective questions |
| **Summarization** | Requesting a summary |
| **Troubleshooting** | Problem-solving questions |
| **Non-Question** | Statements or acknowledgments |

#### Examples from the benchmark:

```json
{
  "text": "How many teams are in the NFL?",
  "enrichments": {
    "Question Type": ["Factoid"]
  }
}
```

```json
{
  "text": "Are the Arizona Cardinals and the Chicago Cardinals the same team?",
  "enrichments": {
    "Question Type": ["Explanation"]
  }
}
```

---

### 3. Multi-Turn (3 categories)

Indicates relationship to previous conversation turns.

| Category | Description | Example |
|----------|-------------|---------|
| **Follow-up** | Builds on previous answer, moves conversation forward | After discussing Cardinals: "How many teams are in the NFL?" |
| **Clarification** | Asks for more detail about previous topic | "Are the Arizona Cardinals and the Chicago Cardinals the same team?" |
| **N/A** | First turn of conversation (no previous context) | Initial question in a conversation |

---

## Complete Example

Here's how enrichments appear in an actual conversation message:

```json
{
  "speaker": "user",
  "text": "Do the Arizona Cardinals play outside the US?",
  "enrichments": {
    "Question Type": ["Explanation"],
    "Multi-Turn": ["Clarification"],
    "Answerability": ["ANSWERABLE"]
  }
}
```

---

## Why Enrichments Matter

1. **Performance Analysis**: Allows measuring model performance by question type, answerability, and multi-turn category
   - Example: "Do models struggle more with unanswerable questions?"
   - Example: "How well do models handle clarification questions vs follow-ups?"

2. **Challenge Design**: Ensures diverse questions across the benchmark
   - Not all questions are simple factoids
   - Includes challenging unanswerable questions
   - Mix of follow-ups and clarifications

3. **Fine-grained Evaluation**: Break down results to identify specific weaknesses
   - "Model X is good at factoid questions but fails on composite questions"
   - "All models struggle with unanswerable questions (hallucination)"

4. **Real-world Simulation**: Natural conversations have this variety
   - People ask different types of questions
   - Not everything is answerable
   - Conversations involve follow-ups and clarifications

5. **Research Insights**: The paper uses enrichments extensively to analyze:
   - Which question types are most challenging for RAG systems
   - How answerability affects model behavior (especially hallucinations on unanswerable questions)
   - Whether multi-turn dependencies cause performance degradation

---

## Statistics from the Benchmark

The benchmark contains diverse distributions across all enrichment categories:

- **Answerability**: Mix of answerable, unanswerable, partial, and conversational questions
- **Question Types**: All 10 types represented, ensuring comprehensive evaluation
- **Multi-Turn**: Progressive mix of N/A (first turns), follow-ups, and clarifications as conversations develop

This diversity makes mtRAG a comprehensive test of RAG system capabilities across realistic conversation scenarios.

