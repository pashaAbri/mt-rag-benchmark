# Hybrid Extractive Query Rewriting - Implementation Details
## Approach 2: MMR + Templates + Entity Recognition

## Overview

This approach enhances pure MMR-based extraction with linguistic features and structural templates:

1. **Question Templates** - Provide grammatical structure
2. **Entity Recognition** - Identify and prioritize named entities
3. **Question Type Classification** - Determine question category (What/Where/How)
4. **Pronoun Detection** - Attempt basic resolution using entity matching

**Core Enhancement:** Combines term selection with template-based structure while remaining extractive in nature.

---

## Architecture

```
┌─────────────────────────┐
│  Current Query +        │
│  Conversation History   │
└──────────┬──────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────┐  ┌───────────────┐
│ Question │  │ Entity        │
│ Type     │  │ Recognition   │
│ Classifier│  │ (spaCy)       │
└──────────┘  └───────────────┘
      │             │
      └──────┬──────┘
             ▼
   ┌──────────────────┐
   │ MMR Selection    │
   │ (with entity     │
   │  boosting)       │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ Template         │
   │ Application      │
   │ - Choose template│
   │ - Fill slots     │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ Post-Processing  │
   │ - Capitalize     │
   │ - Add ?          │
   │ - Clean          │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ Rewritten Query  │
   │ (structured)     │
   └──────────────────┘
```

---

## Implementation

**Full implementation:** `scripts/ideas/retrieval_tasks/hybrid_extractive/hybrid_extractive_rewrite.py`

### Core Components

**HybridExtractiveRewriter Class:**
- `__init__`: Initialize with base MMR rewriter, spaCy NLP, entity boost parameter
- `rewrite()`: Main rewriting function with 6-step pipeline
- `classify_question_type()`: Detect question category (10 types)
- `extract_entities()`: Extract named entities using spaCy NER
- `detect_pronouns()`: Identify pronouns with POS tagging
- `mmr_with_entity_boost()`: Enhanced MMR prioritizing entities
- `apply_template()`: Apply question-type-specific templates
- `post_process()`: Clean and format final output

### Key Features

**Question Type Classification:**
- 10 supported types: what, where, when, who, why, how, is, do, keyword, statement
- First-word pattern matching
- Structural analysis for implicit questions

**Entity Recognition (spaCy):**
- Extracts PERSON, ORG, GPE, PRODUCT, FAC, LOC entities
- Tracks entity source (query vs history)
- Deduplication with recency prioritization

**Entity Boosting in MMR:**
- Multiplies relevance scores by boost factor (default 1.5x)
- Ensures entities are prioritized in term selection
- Helps preserve important named entities

**Template System:**
- Question-type-specific templates
- Verb tense selection heuristics
- Slot filling with entities and terms
- Examples:
  - `what`: "What is/was [terms]?"
  - `where`: "Where [terms]?"
  - `who`: "Who is/was [terms]?"
  - `keyword`: "What about [terms]?"

**Post-Processing:**
- Removes duplicate consecutive words
- Capitalizes first letter
- Adds question mark
- Trims whitespace

### Usage

```python
from hybrid_extractive_rewrite import HybridExtractiveRewriter
from pure_extractive_rewrite import load_mtrag_queries

# Initialize
rewriter = HybridExtractiveRewriter(
    lambda_param=0.7,
    max_terms=10,
    entity_boost=1.5
)

# Load and rewrite
queries = load_mtrag_queries('clapnq')
for item in queries:
    rewritten = rewriter.rewrite(item['query'], item['history'])
```

**Command line:**
```bash
python scripts/ideas/retrieval_tasks/hybrid_extractive/hybrid_extractive_rewrite.py clapnq
```

---

## Expected Output Examples

### Example 1: Pronoun Resolution

```
Original: "what did he do?"
History: ["name the viceroy at the time of quit india movement"]

Pure Extractive Output:    "viceroy quit india movement"
Hybrid Extractive Output:  "What did viceroy quit india movement?"

Comparison: Hybrid adds question structure ("What did") but fails to resolve 
           "he" to "Lord Linlithgow" and produces ungrammatical output
```

### Example 2: Keyword Expansion

```
Original: "population back then"
History: ["quit india movement"]

Pure Extractive Output:    "population quit india movement"
Hybrid Extractive Output:  "What was population quit india movement?"

Comparison: Hybrid adds question structure ("What was") but produces 
           ungrammatical output (missing "the", "of India", "during")
```

### Example 3: Standalone Query

```
Original: "Are the Arizona Cardinals and Chicago Cardinals the same?"
History: ["where do the cardinals play this week"]

Pure Extractive Output:    "arizona cardinals chicago same team week play"
Hybrid Extractive Output:  "Are Arizona Cardinals Chicago Cardinals same?"

Comparison: Hybrid preserves question structure better but may still add noise
```

---

## Known Limitations

### 1. Pronoun Resolution Remains Imperfect

Simple entity matching frequently selects incorrect antecedent. For queries with multiple entities of the same type, recency heuristic often fails.

Example: "what did he do?" may resolve "he" to the wrong person when multiple PERSON entities exist in history.

### 2. Template Rigidity

One-size-fits-all templates cannot capture linguistic nuance. Templates produce grammatically incomplete output (missing articles, prepositions, proper verb forms).

Example: "What was population quit india movement?" instead of "What was the population of India during the Quit India Movement?"

### 3. Noise Addition

MMR may still select irrelevant terms from conversation history, even with entity boosting. Standalone queries with conversation context remain at risk.

### 4. No Semantic Inference

Cannot infer implicit connections or paraphrase concepts. Direct term concatenation limits semantic understanding.

Examples:
- Cannot infer "regardless of location" from "outside US"
- Cannot connect "back then" to specific temporal context
