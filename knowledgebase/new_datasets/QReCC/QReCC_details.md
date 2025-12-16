# QReCC Dataset Information

## Overview

QReCC (**Q**uestion **Re**writing in **C**onversational **C**ontext) is an end-to-end open-domain question answering dataset designed for conversational question answering tasks. The dataset includes question rewriting, passage retrieval, and reading comprehension subtasks.

- **Paper**: [Open-Domain Question Answering Goes Conversational via Question Rewriting](https://arxiv.org/abs/2010.04898)
- **Task**: Find answers to conversational questions within a collection of 10M web pages split into 54M passages
- **License**: Creative Commons Attribution-ShareAlike 3.0 Unported License

---

## Dataset Statistics

### Overall Statistics

| Metric                   | Value                  |
| ------------------------ | ---------------------- |
| **Total Entries**        | 79,952                 |
| **Unique Conversations** | 13,598                 |
| **Train Split**          | 63,501 entries (79.4%) |
| **Test Split**           | 16,451 entries (20.6%) |

### Train Split Statistics

| Metric                             | Value  |
| ---------------------------------- | ------ |
| **Total Entries**                  | 63,501 |
| **Unique Conversations**           | 10,823 |
| **Average Turns per Conversation** | 5.87   |
| **Median Turns per Conversation**  | 5.0    |
| **Min Turns per Conversation**     | 1      |
| **Max Turns per Conversation**     | 12     |

**Conversation Sources:**

- **QuAC**: 50,360 entries (79.31%)
- **Natural Questions**: 13,141 entries (20.69%)

### Test Split Statistics

| Metric                             | Value  |
| ---------------------------------- | ------ |
| **Total Entries**                  | 16,451 |
| **Unique Conversations**           | 2,775  |
| **Average Turns per Conversation** | 5.93   |
| **Median Turns per Conversation**  | 5.0    |
| **Min Turns per Conversation**     | 1      |
| **Max Turns per Conversation**     | 12     |

**Conversation Sources:**

- **QuAC**: 12,389 entries (75.31%)
- **Natural Questions**: 3,314 entries (20.14%)
- **TREC CAsT**: 748 entries (4.55%)

---

## Text Length Statistics

### Train Split (in words)

| Field                         | Mean  | Median | Min | Max |
| ----------------------------- | ----- | ------ | --- | --- |
| **Question**                  | 6.58  | 6.0    | 1   | 28  |
| **Rewrite**                   | 10.10 | 9.0    | 1   | 89  |
| **Answer**                    | 16.88 | 18.0   | 0   | 71  |
| **Context (total words)**     | 79.72 | 67.0   | 0   | 466 |
| **Context (number of items)** | 5.78  | 6.0    | 0   | 22  |

### Test Split (in words)

| Field                         | Mean  | Median | Min | Max |
| ----------------------------- | ----- | ------ | --- | --- |
| **Question**                  | 6.50  | 6.0    | 1   | 21  |
| **Rewrite**                   | 9.92  | 9.0    | 2   | 88  |
| **Answer**                    | 17.66 | 18.0   | 0   | 113 |
| **Context (total words)**     | 83.26 | 69.0   | 0   | 561 |
| **Context (number of items)** | 5.83  | 6.0    | 0   | 22  |

### Key Observations

- **Questions** are typically short (~6-7 words), reflecting natural conversational queries
- **Rewrites** are longer (~10 words) as they resolve context dependencies (anaphora, ellipsis)
- **Answers** average ~17 words, providing concise responses
- **Context** contains ~6 previous Q&A pairs on average, building conversation history

---

## Data Structure

### Entry Format

Each entry in the dataset is a JSON object with the following fields:

```json
{
  "Conversation_no": 1,
  "Turn_no": 2,
  "Question": "Did Gary sing well?",
  "Context": [
    "What can you tell me about Gary Cherone?",
    "Gary Francis Caine Cherone is an American rock singer..."
  ],
  "Rewrite": "Did Gary Cherone sing well?",
  "Answer": "Yes, Gary Cherone is also known for his work...",
  "Answer_URL": "https://en.wikipedia.org/wiki/Van_Halen",
  "Conversation_source": "quac"
}
```

### Field Descriptions

| Field                 | Type           | Description                                           | Coverage |
| --------------------- | -------------- | ----------------------------------------------------- | -------- |
| `Conversation_no`     | `int`          | Unique identifier for the conversation                | 100%     |
| `Turn_no`             | `int`          | Turn number within the conversation (1-12)            | 100%     |
| `Question`            | `string`       | The original context-dependent question               | 100%     |
| `Context`             | `list[string]` | Previous questions and answers in the conversation    | 100%     |
| `Rewrite`             | `string`       | Context-independent rewritten version of the question | 100%     |
| `Answer`              | `string`       | The answer text for the question                      | ~91%     |
| `Answer_URL`          | `string`       | URL(s) to the web page(s) used to produce the answer  | ~91%     |
| `Conversation_source` | `string`       | Source dataset: `quac`, `nq`, or `trec`               | 100%     |

### Missing Data

- **Answer**: ~8.75% missing in train, ~8.67% missing in test
- **Answer_URL**: ~8.99% missing in train, ~8.97% missing in test

These missing values typically occur when answers cannot be found or URLs are unavailable.

---

## Dataset Sources

The QReCC dataset is built from three sources:

1. **TREC CAsT** (Conversational Assistance Track)

   - Multi-turn conversational dataset
   - Focus on information-seeking conversations
   - Present only in test split (4.55%)

2. **QuAC** (Question Answering in Context)

   - Multi-turn question answering dataset
   - Wikipedia-based conversations
   - Largest source: ~75-79% of entries

3. **Google Natural Questions (NQ)**
   - Originally single-turn questions
   - Converted to conversations with explicit context-dependent questions
   - Balanced for anaphora (co-references) and ellipsis
   - ~20% of entries

---

## Key Features

### Question Rewriting

Each conversational question has a corresponding **rewrite** that:

- Resolves anaphoric references (e.g., "Did Gary sing well?" → "Did Gary Cherone sing well?")
- Completes ellipsis (e.g., "What about Tesla?" → "What about Tesla the car company?")
- Makes the question context-independent for retrieval

### Conversational Context

- Context is stored as a list of alternating questions and answers
- First turn has empty context
- Subsequent turns build conversation history
- Average conversation has ~6 turns

### Answer Annotation

- Answers are extracted from web pages
- Source URLs are provided for verification
- Answers may span multiple passages or documents
- ~91% of entries have annotated answers and URLs

---

## Corpus Information

### Overview

The QReCC corpus consists of **10 million web pages** split into **54 million passages**. This corpus serves as the knowledge base for answering questions in the dataset.

### Corpus Sources

1. **Common Crawl**

   - Large-scale web crawl archive
   - Contains billions of web pages
   - Filtered to relevant pages for the dataset
   - **Note**: Very large (order of tens of TBs)

2. **Wayback Machine**
   - Historical web page archive
   - Pages referenced in `Answer_URL` fields
   - Captured at specific timestamps (November 2019)
   - Smaller subset compared to Common Crawl

### Corpus Structure

#### Document Format

Each document in the corpus is stored in JSONL (JSON Lines) format:

```json
{
  "id": "https://example.com/page",
  "contents": "Full text content of the webpage..."
}
```

#### Passage Format

Documents are chunked into passages with minimum 220 tokens per passage:

```json
{
  "id": "https://example.com/page_p0",
  "contents": "Passage text content..."
}
```

- **Passage ID**: `{document_id}_p{passage_number}`
- **Minimum tokens**: 220 tokens per passage
- **Chunking method**: Line-based with token accumulation

### Corpus Processing Pipeline

1. **Download** web pages from Common Crawl and Wayback Machine
2. **Extract** text content from HTML pages
3. **Chunk** documents into passages (minimum 220 tokens)
4. **Index** passages using Pyserini/Anserini (Lucene-based)

### Corpus Building

The corpus is **not included** in the dataset repository and must be built separately. The repository provides scripts in the `collection/` directory:

- `download_commoncrawl_passages.py`: Downloads pages from Common Crawl
- `download_wayback_passages.py`: Downloads pages from Wayback Machine
- `paragraph_chunker.py`: Chunks documents into passages

**Note**: Building the full corpus requires significant storage space and time.

### Corpus Usage

The corpus is used for:

- **Retrieval**: Finding relevant passages for answering questions
- **Evaluation**: Measuring retrieval and reading comprehension performance
- **Training**: Training retrieval models for conversational QA

---

## Task Description

The QReCC task involves:

1. **Question Rewriting**: Convert context-dependent questions to context-independent queries
2. **Passage Retrieval**: Find relevant passages from the 54M passage corpus
3. **Reading Comprehension**: Extract or generate answers from retrieved passages

Answers to questions in the same conversation may be distributed across several web pages, making the task more challenging than single-turn QA.

---

## Evaluation

The repository provides evaluation scripts:

- **Retrieval QA**: `utils/evaluate_retrieval.py`
- **Extractive QA**: `utils/evaluate_qa.py`

---

## Files

- `dataset/qrecc_train.json`: Training split (63,501 entries)
- `dataset/qrecc_test.json`: Test split (16,451 entries)
- `analyze_dataset.py`: Script for analyzing dataset statistics

---

## Citation

If you use the QReCC dataset, please cite:

```bibtex
@article{qrecc,
  title={Open-Domain Question Answering Goes Conversational via Question Rewriting},
  author={Anantha, Raviteja and Vakulenko, Svitlana and Tu, Zhucheng and Longpre, Shayne and Pulman, Stephen and Chappidi, Srinivas},
  journal={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2021}
}
```

---

## Additional Resources

- **Repository**: [https://github.com/apple/ml-qrecc](https://github.com/apple/ml-qrecc)
- **Paper**: [https://arxiv.org/abs/2010.04898](https://arxiv.org/abs/2010.04898)
- **Original Sources**:
  - [TREC CAsT](https://github.com/daltonj/treccastweb/tree/master/2019/data)
  - [QuAC](https://quac.ai)
  - [Natural Questions](https://github.com/google-research-datasets/natural-questions)
