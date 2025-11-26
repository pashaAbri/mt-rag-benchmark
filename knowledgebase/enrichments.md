# Enrichments in mtRAG

**Enrichments** are metadata annotations that human annotators added to each user question during the conversation
creation process. They classify questions along three key dimensions that are central to the benchmark's design.

---

Initial annotation:

- Three dimensions: Question Type (10 categories), Answerability (4 categories), Multi-Turn Type (3 categories)
- Review Process: Annotators could "repair responses, passage relevance, and dimensions as needed"

## Three Types of Enrichments

### 1. Answerability (4 categories)

Indicates whether the question can be answered given the retrieved passages.

| Category           | Description                                                                   | Example                                                 |
|--------------------|-------------------------------------------------------------------------------|---------------------------------------------------------|
| **ANSWERABLE**     | Can be fully answered from passages                                           | "Do the Arizona Cardinals play outside the US?"         |
| **UNANSWERABLE**   | Cannot be answered from available passages (needs current info not in corpus) | "where do the arizona cardinals play this week"         |
| **PARTIAL**        | Can be partially answered (some info available, but incomplete)               | Questions where corpus has some but not all needed info |
| **CONVERSATIONAL** | Not a factual question; conversational/acknowledgment                         | "Thank you" or "I see"                                  |

---

### 2. Question Type (10 categories)

Classifies the nature of the question.

| Type                | Description                        |
|---------------------|------------------------------------|
| **Factoid**         | Simple fact-based questions        |
| **Explanation**     | Requires detailed explanation      |
| **Composite**       | Multiple questions in one          |
| **Comparative**     | Comparing two or more things       |
| **How-To**          | Procedural/instructional questions |
| **Keyword**         | Keyword-based search queries       |
| **Opinion**         | Subjective questions               |
| **Summarization**   | Requesting a summary               |
| **Troubleshooting** | Problem-solving questions          |
| **Non-Question**    | Statements or acknowledgments      |

---

### 3. Multi-Turn (3 categories)

Indicates relationship to previous conversation turns.

| Category          | Description                                           | Example                                                              |
|-------------------|-------------------------------------------------------|----------------------------------------------------------------------|
| **Follow-up**     | Builds on previous answer, moves conversation forward | After discussing Cardinals: "How many teams are in the NFL?"         |
| **Clarification** | Asks for more detail about previous topic             | "Are the Arizona Cardinals and the Chicago Cardinals the same team?" |
| **N/A**           | First turn of conversation (no previous context)      | Initial question in a conversation                                   |

---

## Enrichment Statistics

Statistics calculated from all 842 tasks in the `cleaned_data/tasks/` directory.

### Question Type Statistics

| Question Type   | Count    | Percentage |
|-----------------|----------|------------|
| Factoid         | 280      | 25.9%      |
| Summarization   | 195      | 18.1%      |
| Explanation     | 158      | 14.6%      |
| Opinion         | 87       | 8.1%       |
| How-To          | 85       | 7.9%       |
| Non-Question    | 84       | 7.8%       |
| Keyword         | 76       | 7.0%       |
| Composite       | 51       | 4.7%       |
| Comparative     | 48       | 4.4%       |
| Troubleshooting | 15       | 1.4%       |
| **Total**       | **1079** | **100.0%** |

*Note: Some tasks may have multiple Question Type labels, which is why the total Question Type labels (1079) exceeds the
total number of tasks (842).*

### Multi-Turn Type Statistics

| Multi-Turn Type | Count   | Percentage |
|-----------------|---------|------------|
| Follow-up       | 622     | 73.9%      |
| N/A             | 110     | 13.1%      |
| Clarification   | 110     | 13.1%      |
| **Total**       | **842** | **100.0%** |

### Answerability Statistics

| Answerability Type | Count   | Percentage |
|--------------------|---------|------------|
| ANSWERABLE         | 709     | 84.2%      |
| PARTIAL            | 68      | 8.1%       |
| UNANSWERABLE       | 55      | 6.5%       |
| CONVERSATIONAL     | 10      | 1.2%       |
| **Total**          | **842** | **100.0%** |

### Domain Statistics

| Domain    | Count   | Percentage |
|-----------|---------|------------|
| CLAPNQ    | 224     | 26.6%      |
| GOVT      | 214     | 25.4%      |
| CLOUD     | 205     | 24.3%      |
| FIQA      | 199     | 23.6%      |
| **Total** | **842** | **100.0%** |