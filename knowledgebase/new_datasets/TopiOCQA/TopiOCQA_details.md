# TopiOCQA Dataset Documentation

This document provides comprehensive information about the TopiOCQA dataset, including statistics, the Wikipedia corpus, and available human annotations.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Dataset Statistics](#dataset-statistics)
3. [Data Structure](#data-structure)
4. [Wikipedia Corpus](#wikipedia-corpus)
5. [Human Annotations](#human-annotations)
6. [Question Representation Variants](#question-representation-variants)

---

## Dataset Overview

**TopiOCQA** (Open-domain Conversational Question Answering with Topic Switching) is a dataset designed for open-domain conversational question answering where conversations can switch topics mid-dialogue. This makes it more challenging than traditional conversational QA datasets that maintain topic consistency.

### Key Characteristics

- **Multi-turn conversations**: Each conversation contains multiple question-answer pairs
- **Topic switching**: Conversations can switch between different topics within the same dialogue
- **Open-domain**: Questions can be about any topic covered in Wikipedia
- **Abstractive answers**: Answers are not necessarily extractive spans but can be abstractive summaries
- **Context accumulation**: Each turn builds upon previous conversation history

---

## Dataset Statistics

### Training Set

| Metric                                | Value         |
| ------------------------------------- | ------------- |
| **Total conversation turns**          | 45,450        |
| **Total conversations**               | 3,509         |
| **Average turns per conversation**    | 12.95         |
| **Conversations with topic switches** | 3,416 (97.3%) |
| **Unique topics**                     | 9,527         |
| **Average question length**           | 6.91 words    |
| **Average answer length**             | 11.96 words   |
| **Unanswerable questions**            | 3,652 (8.0%)  |
| **Questions from Natural Questions**  | 2,487 (5.5%)  |

### Development Set

| Metric                                | Value       |
| ------------------------------------- | ----------- |
| **Total conversation turns**          | 2,514       |
| **Total conversations**               | 205         |
| **Average turns per conversation**    | 12.26       |
| **Conversations with topic switches** | 200 (97.6%) |
| **Unique topics**                     | 734         |
| **Questions from Natural Questions**  | 164 (6.5%)  |
| **Unanswerable questions**            | 174 (6.9%)  |

### Top Topics (Training Set)

The most frequently occurring topics in the training set:

1. **Game of Thrones** - 105 turns
2. **The Lord of the Rings** - 81 turns
3. **New York City** - 71 turns
4. **Grey's Anatomy** - 70 turns
5. **United States** - 68 turns

---

## Data Structure

### Raw Dataset Format

Each entry in the dataset represents a single turn in a conversation. The dataset is stored as a JSON array where each object contains:

```json
{
  "Conversation_no": 1,
  "Turn_no": 1,
  "Question": "what was australia's contribution to the battle of normandy?",
  "Answer": "The army personnel and thousands of Australian airmen took part in the battle.",
  "Topic": "Australian contribution to the Battle of Normandy",
  "Topic_section": "Introduction",
  "Context": [],
  "Rationale": "The army personnel and thousands of Australian airmen also took part in the subsequent Battle of Normandy between June and August 1944, and an RAAF fighter squadron operated from airfields in Normandy.",
  "is_nq": false
}
```

### Field Descriptions

| Field             | Type         | Description                                                                             |
| ----------------- | ------------ | --------------------------------------------------------------------------------------- |
| `Conversation_no` | Integer      | Unique identifier for the conversation                                                  |
| `Turn_no`         | Integer      | Turn number within the conversation (starts at 1)                                       |
| `Question`        | String       | The question asked in this turn                                                         |
| `Answer`          | String       | The answer to the question (or "UNANSWERABLE" if no answer exists)                      |
| `Topic`           | String       | The topic/subject of this turn                                                          |
| `Topic_section`   | String       | Section within the topic (e.g., "Introduction", "History")                              |
| `Context`         | List[String] | List of previous question-answer pairs in the conversation (accumulates with each turn) |
| `Rationale`       | String       | Supporting text/passage that justifies the answer                                       |
| `is_nq`           | Boolean      | Whether the question originates from the Natural Questions dataset                      |

### Example Conversation Flow

Here's an example showing how a conversation evolves:

**Turn 1:**

- Question: "what was australia's contribution to the battle of normandy?"
- Answer: "The army personnel and thousands of Australian airmen took part in the battle."
- Topic: "Australian contribution to the Battle of Normandy"
- Context: `[]` (empty for first turn)

**Turn 2:**

- Question: "was the battle fought in australia?"
- Answer: "UNANSWERABLE"
- Topic: "Australian contribution to the Battle of Normandy"
- Context: `["what was australia's contribution to the battle of normandy?", "The army personnel..."]`

**Turn 7:** (Topic switch occurs)

- Question: "when did the lieutenant begin his military career?"
- Answer: "During the pre-war years."
- Topic: "Ronald McNicoll" (topic switched from previous turns)
- Context: `[...previous 12 Q-A pairs...]`

---

## Wikipedia Corpus

### Overview

The Wikipedia corpus serves as the knowledge base for retrieval-based question answering. It contains the full Wikipedia dump used for training and evaluation.

### Corpus Statistics

| Metric                     | Value                        |
| -------------------------- | ---------------------------- |
| **Total articles**         | 5,824,983                    |
| **File size**              | ~9.9 GB (compressed)         |
| **Format**                 | JSONL (one article per line) |
| **Average article length** | ~1,166 characters            |

### Article Structure

Each Wikipedia article in the corpus contains the following fields:

```json
{
  "title": "Article Title",
  "text": "Full article text content...",
  "links": ["list", "of", "hyperlinks"],
  "ners": ["named", "entities"],
  "nouns": ["extracted", "nouns"],
  "verbs": ["extracted", "verbs"],
  "adjs": ["extracted", "adjectives"],
  "propn": ["proper", "nouns"]
}
```

### Field Descriptions

| Field   | Type         | Description                               |
| ------- | ------------ | ----------------------------------------- |
| `title` | String       | Wikipedia article title                   |
| `text`  | String       | Full article text content                 |
| `links` | List[String] | Hyperlinks found within the article       |
| `ners`  | List[String] | Named entities extracted from the article |
| `nouns` | List[String] | Nouns extracted from the article          |
| `verbs` | List[String] | Verbs extracted from the article          |
| `adjs`  | List[String] | Adjectives extracted from the article     |
| `propn` | List[String] | Proper nouns extracted from the article   |

### Segmented Passages

For modeling purposes, Wikipedia articles are chunked into passages:

- **Format**: TSV (Tab-Separated Values)
- **Structure**: `id`, `text`, `title` (tab-separated)
- **Chunking method**: Articles split into 200-word segments with no overlap
- **Total segments**: ~25.7 million passages

Each segment contains:

- **Segment ID**: Unique identifier
- **Text**: Passage content (~200 words)
- **Title**: Source article title

---

## Human Annotations

The TopiOCQA dataset includes several types of human annotations:

### 1. Gold Passages

**Purpose**: Gold passages are Wikipedia passages that contain the answer to a question. These are used for:

- Training retrieval models
- Evaluating retrieval performance
- Providing ground truth for reader models

**Availability**: Gold passage annotations are available for all three question representation variants:

- `all_history` - Questions include full conversation history
- `original` - Questions without context
- `rewrites_t5_qrecc` - Questions rewritten to be self-contained

**Format**: Each gold passage entry contains:

```json
{
  "question": "question text",
  "title": "Wikipedia article title",
  "context": "passage text containing the answer"
}
```

**Download**:

```bash
# All history variant
python download_data.py --resource data.gold_passages_info.all_history.train
python download_data.py --resource data.gold_passages_info.all_history.dev

# Original variant
python download_data.py --resource data.gold_passages_info.original.train
python download_data.py --resource data.gold_passages_info.original.dev

# Rewrites variant
python download_data.py --resource data.gold_passages_info.rewrites_t5_qrecc.train
python download_data.py --resource data.gold_passages_info.rewrites_t5_qrecc.dev
```

### 2. Topic Annotations

Each turn is annotated with:

- **Topic**: The main topic/subject of the question
- **Topic_section**: The specific section within the topic (e.g., "Introduction", "History", "Characters")

These annotations enable:

- Analysis of topic distribution
- Evaluation of topic switching behavior
- Topic-aware model training

### 3. Answer Quality Annotations

- **Answerable vs. Unanswerable**: Questions are marked as answerable or "UNANSWERABLE"
- **Rationale**: Each answer includes a rationale passage that supports the answer
- **Answer source**: Some questions are marked with `is_nq` flag indicating they come from Natural Questions dataset

### 4. Conversation Context

The `Context` field provides:

- **Accumulated history**: Previous question-answer pairs in the conversation
- **Context length**: Grows linearly with each turn (Turn N has 2\*(N-1) context items)
- **Enables**: Context-aware question understanding and answer generation

### 5. Retriever Training Data

Pre-processed data formatted for DPR (Dense Passage Retrieval) training:

**Format**:

```json
{
  "question": "question text",
  "answers": ["answer1", "answer2", ...],
  "positive_ctxs": [{
    "title": "article title",
    "text": "passage text"
  }],
  "negative_ctxs": [...],
  "hard_negative_ctxs": [...]
}
```

**Available variants**:

- `data.retriever.all_history.train/dev`
- `data.retriever.original.train/dev`
- `data.retriever.rewrites_t5_qrecc.train/dev`

**CSV format** (for inference):

- `data.retriever.qas.all_history.train/dev`
- `data.retriever.qas.original.train/dev`
- `data.retriever.qas.rewrites_t5_qrecc.train/dev`

---

## Question Representation Variants

TopiOCQA provides three question representation variants for different modeling approaches:

### 1. Original

**Description**: Uses only the current question without any conversation history.

**Example**:

- Question: "was it nominated for any award?"

**Use case**: Baseline comparison, simple question answering without context.

### 2. AllHistory

**Description**: Includes full conversation history with question-answer pairs separated by `[SEP]` tokens.

**Example**:

- Question: "who is lead singer of rage against the machine [SEP] Zack de la Rocha [SEP] when was it formed [SEP] 1991 [SEP] was it nominated for any award?"

**Use case**: Models that can leverage full conversation context.

### 3. Rewrites (T5-QReCC)

**Description**: Questions are rewritten to be self-contained using T5-QReCC model, incorporating necessary context from conversation history.

**Example**:

- Question: "was rage against the machine nominated for any award?"

**Use case**: Models that benefit from self-contained questions but want to leverage conversation context.

---

## Download Instructions

### Basic Dataset Files

```bash
# Training and development sets
python download_data.py --resource data.topiocqa_dataset.train
python download_data.py --resource data.topiocqa_dataset.dev
```

### Wikipedia Corpus

```bash
# Full Wikipedia articles (large file ~9.9 GB)
python download_data.py --resource data.wikipedia_split.full_wiki

# Segmented passages (large file ~several GB)
python download_data.py --resource data.wikipedia_split.full_wiki_segments
```

### Gold Passages

```bash
# All history variant
python download_data.py --resource data.gold_passages_info.all_history

# Original variant
python download_data.py --resource data.gold_passages_info.original

# Rewrites variant
python download_data.py --resource data.gold_passages_info.rewrites_t5_qrecc
```

### Retriever Training Data

```bash
# All history variant (JSON format)
python download_data.py --resource data.retriever.all_history

# CSV format for inference
python download_data.py --resource data.retriever.qas.all_history
```

---

## Dataset Characteristics Summary

### Strengths

1. **Realistic conversations**: Multi-turn dialogues that mirror real-world interactions
2. **Topic switching**: 97%+ of conversations contain topic switches, making it more challenging
3. **Diverse topics**: Over 9,500 unique topics in training set
4. **Rich annotations**: Gold passages, topics, rationales, and context information
5. **Large scale**: 45K+ training turns, 2.5K+ dev turns
6. **Open-domain**: Questions span all Wikipedia topics

### Challenges

1. **Context dependency**: Questions often require understanding previous turns
2. **Topic switching**: Models must track and adapt to topic changes
3. **Abstractive answers**: Answers may not be direct extracts from passages
4. **Unanswerable questions**: ~8% of questions have no answer
5. **Large corpus**: ~5.8M Wikipedia articles require efficient retrieval

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{adlakha2022topiocqa,
  title={Topi{OCQA}: Open-domain Conversational Question Answering with Topic Switching},
  author={Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  journal={Transactions of the Association for Computational Linguistics},
  volume = {10},
  pages = {468-483},
  year = {2022},
  month = {04},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00471},
  url = {https://doi.org/10.1162/tacl_a_00471}
}
```

---

## Additional Resources

- **Project Page**: https://mcgill-nlp.github.io/topiocqa/
- **Paper**: https://arxiv.org/abs/2110.00768
- **Repository**: This repository contains code for DPR and FiD models

---

_Last updated: Based on dataset version available as of December 2024_
