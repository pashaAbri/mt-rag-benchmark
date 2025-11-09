# Pure Extractive Query Rewriting - Implementation Details
## Approach 1: MMR-Based Term Selection

## Overview

This approach uses Maximal Marginal Relevance (MMR) to select relevant, non-redundant terms from the conversation history and current query. MMR represents the best-case scenario for extractive methods, balancing relevance with diversity to avoid redundancy.

**Core Principle:** Select and combine existing terms without generating new text.

---

## Method: Maximal Marginal Relevance (MMR)

### MMR Formula

```
MMR(Di) = λ·sim(Di, Q) - (1-λ)·max[sim(Di, Dj)]
                                   Dj∈S
```

Where:
- `Di` = candidate term/phrase
- `Q` = current query + context
- `S` = already selected terms
- `λ` = relevance vs diversity balance (typically 0.7)
- `sim()` = similarity function (cosine similarity of embeddings)

**Goal:** Select terms that are:
1. **Relevant** to the current query
2. **Novel** (not redundant with already-selected terms)

---

## Architecture

```
┌────────────────────────┐
│  Current Query +       │
│  Conversation History  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Text Preprocessing    │
│  - Tokenization        │
│  - Stop word removal   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Candidate Extraction  │
│  - N-grams (1-3)       │
│  - Named entities      │
│  - Key phrases         │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  MMR Selection         │
│  - Compute embeddings  │
│  - Select iteratively  │
│  - Balance λ           │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Term Concatenation    │
│  - Join selected terms │
│  - Remove duplicates   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Rewritten Query       │
│  (keyword sequence)    │
└────────────────────────┘
```

---

## Implementation

**Full implementation:** `scripts/ideas/retrieval_tasks/pure_extractive/pure_extractive_rewrite.py`

### Core Components

**PureExtractiveRewriter Class:**
- `__init__`: Initialize with sentence embedding model, lambda parameter, max terms
- `rewrite()`: Main rewriting function
- `extract_candidates()`: Generate unigrams, bigrams, trigrams, and named entities
- `mmr_select()`: Select k terms using MMR algorithm
- `format_history()`: Convert conversation history to text

### Key Functions

**Text Preprocessing:**
- Tokenization using NLTK
- Stop word removal (preserving question words)
- Lowercasing and special character handling

**Candidate Extraction:**
- N-gram generation (1-3 words)
- Optional spaCy NER integration
- Deduplication while preserving order

**MMR Selection:**
- Iterative term selection balancing relevance and diversity
- Cosine similarity for relevance scoring
- Redundancy penalty for diversity

### Usage

```python
from pure_extractive_rewrite import PureExtractiveRewriter, load_mtrag_queries

# Initialize
rewriter = PureExtractiveRewriter(lambda_param=0.7, max_terms=10)

# Load queries
queries = load_mtrag_queries('clapnq')

# Rewrite
for item in queries:
    rewritten = rewriter.rewrite(item['query'], item['history'])
```

**Command line:**
```bash
python scripts/ideas/retrieval_tasks/pure_extractive/pure_extractive_rewrite.py clapnq
```

---

## Expected Output Examples

### Example 1: Standalone Query

```
Original: "Are the Arizona Cardinals and the Chicago Cardinals the same team?"
History: ["where do the arizona cardinals play this week",
          "Do the Arizona Cardinals play outside the US?"]

MMR Output: "arizona cardinals chicago cardinals same team"

Analysis: Removes question structure and grammaticality. May also add noise from 
         history ("play", "week") despite query being already standalone.
```

### Example 2: Pronoun Resolution

```
Original: "what did he do?"
History: ["name the viceroy at the time of quit india movement"]

MMR Output: "viceroy quit india movement"

Analysis: Fails to resolve "he" to specific entity (Lord Linlithgow). 
         Produces keyword sequence without question structure.
```

### Example 3: Keyword Expansion

```
Original: "population back then"
History: ["name the viceroy at the time of quit india movement"]

MMR Output: "population quit india movement"

Analysis: Extracts relevant keywords but cannot generate question structure.
         Missing question word ("What"), verb ("was"), and grammatical markers.
```

### Example 4: Context Addition

```
Original: "How many teams are in the NFL playoffs?"
History: ["How many teams are in the NFL?"]

MMR Output: "teams nfl playoffs"

Analysis: Successfully extracts key terms from both turns but loses question 
         structure. Terms are relevant but output is ungrammatical.
```