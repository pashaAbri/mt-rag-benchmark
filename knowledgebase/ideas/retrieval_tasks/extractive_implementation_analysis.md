# Extractive Query Rewriting: Implementation Analysis
## Testing Extractive Approaches for Multi-Turn Conversational Retrieval

**Date:** November 2025  
**Context:** MT-RAG Benchmark Query Rewriting  
**Dataset:** 777 queries across 4 domains (ClapNQ, Cloud, FiQA, Govt)

---

## Executive Summary

This document presents an analysis of 777 query rewrites from the MT-RAG benchmark. The analysis identifies distinct rewriting patterns and their characteristics to inform extractive query rewriting implementation.

**Key Findings:**
- 27% of queries are already standalone and require no rewriting
- 73% of queries require rewriting across five distinct patterns
- Patterns identified: pronoun resolution (17%), keyword expansion (14%), context addition (31%), complex rewrite (10%), minimal paraphrase (1%)

**Implementation Approaches:**
- **Approach 1:** Pure Extractive using MMR-based term selection
- **Approach 2:** Hybrid Extractive combining MMR with templates and entity recognition

---

## Dataset Analysis

### Query Distribution by Domain

| Domain | Total Queries | Avg Turns | Avg Query Length |
|--------|--------------|-----------|------------------|
| ClapNQ | 208 | 7.7 | 8.6 words |
| Cloud | 188 | 7.2 | 9.3 words |
| FiQA | 180 | 6.8 | 7.4 words |
| Govt | 201 | 7.5 | 8.9 words |
| **Total** | **777** | **7.3** | **8.5 words** |

### Conversation Context

**Critical Observation:** At inference time, we cannot know in advance which queries need rewriting. Any rewriting method must be applied to ALL queries without peeking at gold labels.

**Context Distribution:**
- Queries with no conversation history: 13.1% (102 queries)
- Queries with 1-5 prior turns: 45.3% (352 queries)
- Queries with 6+ prior turns: 41.6% (323 queries)

**Implication:** Most queries (86.9%) have conversation context that may or may not be relevant to rewriting.

---

## Pattern Analysis

### Pattern Distribution (Aggregate across 4 domains)

Based on analysis of gold rewrites compared to last turn:

```
Context Addition        ████████████████████████████████ 31.0% (241 queries)
Exact Copy              ████████████████████████████     27.0% (210 queries)
Pronoun Resolution      ████████████████                 16.7% (130 queries)
Keyword Expansion       █████████████                    13.5% (105 queries)
Complex Rewrite         ██████████                       10.4% (81 queries)
Minimal Paraphrase      █                                 1.3% (10 queries)
```

### Pattern Definitions

#### 1. Exact Copy (27.0%)
**Definition:** Gold rewrite is identical to last turn  
**Interpretation:** Query is already standalone and clear

**Examples:**
```
Query: "Are the Arizona Cardinals and the Chicago Cardinals the same team?"
Gold: "Are the Arizona Cardinals and the Chicago Cardinals the same team?"
```

```
Query: "What is the capital of France?"
Gold: "What is the capital of France?"
```

**Note:** 108 of these 210 queries (51%) have conversation history present, but the query doesn't need context from it.

#### 2. Context Addition (31.0%)
**Definition:** Rewrite adds significant context (3+ new words) from conversation history

**Examples:**
```
Turn 1: "where do the arizona cardinals play this week"
Last: "Do the Arizona Cardinals play outside the US?"
Gold: "Where do the Arizona Cardinals play, regardless of location, this week?"
Analysis: Merged concepts from both turns
```

```
Turn 1: "What are the sheltered rooms designated for use?"
Last: "What items should I keep?"
Gold: "What items should I keep in the safe room?"
Analysis: Added location context from Turn 1
```

#### 3. Pronoun Resolution (16.7%)
**Definition:** Last turn contains pronouns (he, she, it, they, etc.)

**Examples:**
```
Previous: "name the viceroy at the time of quit india movement"
Last: "what did he do?"
Gold: "What actions did Lord Linlithgow take?"
Analysis: Resolved "he" → "Lord Linlithgow", rephrased "do" → "actions take"
```

```
Previous: "What is ground water contamination?"
Last: "Can it be clean up?"
Gold: "Can groundwater contamination be cleaned up?"
Analysis: Resolved "it" → "groundwater contamination", fixed grammar
```

#### 4. Keyword Expansion (13.5%)
**Definition:** Last turn is ≤5 words without question structure

**Examples:**
```
Previous: "name the viceroy at the time of quit india movement"
Last: "population back then"
Gold: "What was the population of India during the Quit India Movement?"
Analysis: Added question word, verb, entities from context
```

```
Previous: "Was Britain relying on Southern cotton?"
Last: "Modern cotton gins."
Gold: "Can you tell me about modern cotton gins?"
Analysis: Converted keyword phrase into proper question
```

#### 5. Complex Rewrite (10.4%)
**Definition:** Significant restructuring or paraphrasing

**Examples:**
```
Last: "That is too bad. Although the movement was non-violent, some ended up in violence, right?"
Gold: "Although the movement was non-violent, did some instances of violence still occur?"
Analysis: Removed discourse markers, formalized language
```

---

## Domain-Specific Patterns

### Pattern Distribution by Domain

| Domain | Exact Copy | Pronoun | Keyword | Context | Complex |
|--------|-----------|---------|---------|---------|---------|
| ClapNQ | 27.4% | 22.1% | 8.2% | 28.8% | 13.0% |
| Cloud | 33.5% | 11.7% | 16.0% | 28.2% | 9.0% |
| FiQA | 23.3% | 22.2% | 15.0% | 30.6% | 6.1% |
| Govt | 23.9% | 10.9% | 15.4% | 36.3% | 12.9% |

### Domain Characteristics

**ClapNQ (Wikipedia Q&A):**
- Highest pronoun usage (22.1%)
- Conversational nature leads to entity references
- Historical/factual queries need temporal context

**Cloud (Technical Documentation):**
- Highest exact copy rate (33.5%)
- Technical queries tend to be more precise
- High keyword expansion (16.0%) - users ask with technical terms

**FiQA (Financial Q&A):**
- High pronoun usage (22.2%)
- Very conversational discussions
- Investment/financial concepts heavily referenced

**Govt (Government/Policy):**
- Highest context addition (36.3%)
- Policy questions span multiple related topics
- Complex dependencies between concepts

---

## Extractive Rewriting Challenges

### Challenge 1: Detecting Which Queries Need Rewriting

**Problem:** At inference time, we cannot determine which queries need rewriting without already having the gold rewrites. The rewriting method must be applied uniformly to all queries.

**Key Insight:**
- 27% of queries require no rewriting (exact copies)
- Among these, 51% have conversation history present
- Risk: Extractive methods may contaminate already-good queries by adding irrelevant historical terms

**Example Risk Scenario:**
```
Query: "Are the Arizona Cardinals and the Chicago Cardinals the same team?"
History: ["where do the arizona cardinals play this week",
          "Do the Arizona Cardinals play outside the US?"]

Risk: Term selection method might add "play", "week", "outside", "US" 
      even though query is already standalone
```

### Challenge 2: Pronoun Resolution (16.7% of queries)

**Problem:** Need to resolve pronouns to specific entities

**Data Analysis:**
- "he/she/him/her" → Must resolve to PERSON entity
- "it/its" → Must resolve to ORG, PRODUCT, GPE, concept
- "they/them" → Must resolve to plural entity or organization

**Extractive Capabilities:**
- Entity extraction from conversation history using NER
- Basic pronoun-to-entity matching using recency heuristics

**Limitations:**
- Entity matching frequently selects incorrect antecedent
- No grammatical adjustment after pronoun substitution
- Cannot capture semantic relationships between pronoun and referent

### Challenge 3: Keyword Expansion (13.5% of queries)

**Problem:** Incomplete queries need question structure

**Data Analysis:**
- Average keyword query: 2.4 words
- Needs: question word (What/Where/How), verb, potentially entities from context

**Extractive Capabilities:**
- Identification of short queries (≤5 words)
- Keyword extraction from conversation history

**Limitations:**
- No generation of question words (What/Where/How)
- No verb selection or insertion
- Requires template-based construction for grammatical output

### Challenge 4: Context Addition (31.0% of queries)

**Problem:** Need to merge concepts from multiple turns

**Data Analysis:**
- Average new words added: 4.2 words
- Often requires semantic understanding (e.g., "outside US" → "regardless of location")

**Extractive Capabilities:**
- Term selection from multiple conversation turns
- Redundancy reduction through MMR

**Limitations:**
- Simple concatenation rather than semantic merging
- Cannot infer implicit connections between concepts
- No understanding of discourse-level coherence

---

## Two Extractive Approaches

### Why Two Approaches?

We implement two extractive approaches to test the full spectrum of extractive capabilities:

**Approach 1: Pure Extractive (MMR)**
- Uses best-case term selection without linguistic enhancement
- Establishes baseline for extractive methods
- Tests whether sophisticated term selection alone is sufficient

**Approach 2: Hybrid Extractive (MMR + Templates + NER)**
- Adds grammatical structure through templates
- Incorporates entity recognition and boosting
- Tests whether the primary limitation is lack of structure
- Represents the upper bound of extractive method capability

### Approach 1: Pure Extractive (MMR)

**Method:** Maximal Marginal Relevance (MMR) term selection

**Formula:**
```
MMR(Di) = λ·sim(Di, Q) - (1-λ)·max[sim(Di, Dj)]
                                   Dj∈S
```

**Process:**
1. Extract candidate terms (unigrams, bigrams, trigrams, entities)
2. Select terms that are:
   - Relevant to current query
   - Novel (not redundant with already-selected terms)
3. Concatenate selected terms

**Expected Strengths:**
- Good at selecting relevant keywords
- Reduces redundancy
- Fast execution

**Expected Weaknesses:**
- No grammatical structure
- Cannot resolve pronouns intelligently
- May add noise to standalone queries
- No question word selection

**Target Patterns:**
- ✅ Might help: Context addition (if just needs keywords)
- ⚠️ Risky: Exact copy (might add noise)
- ❌ Likely fails: Pronoun resolution, keyword expansion

### Approach 2: Hybrid Extractive (MMR + Templates)

**Method:** Enhanced MMR with linguistic features

**Components:**
1. **Question Type Classifier** - Detect What/Where/When/How
2. **Entity Recognition** - spaCy NER for person/org/location
3. **MMR with Entity Boosting** - Prioritize entities in selection
4. **Question Templates** - Add grammatical structure
5. **Pronoun Resolution** - Basic entity matching

**Process:**
1. Classify question type from last turn
2. Extract entities from query + history
3. Detect pronouns in query
4. Run MMR with entity boosting
5. Apply question template
6. Post-process (capitalize, add ?)

**Expected Strengths:**
- Better grammaticality than pure extractive
- Entity preservation
- Basic pronoun handling
- Question structure

**Expected Weaknesses:**
- Template rigidity (one-size-fits-all)
- Pronoun resolution often picks wrong entity
- Still may add noise
- Cannot infer implicit meaning

**Target Patterns:**
- ✅ Should help: Keyword expansion (adds structure)
- ⚠️ Might help: Pronoun resolution (if entity matching works)
- ⚠️ Better than pure: Context addition (structured concatenation)
- ⚠️ Still risky: Exact copy (less noise but still some)

---

## Appendix: Data Summary

### Query Length Distribution

| Length | ClapNQ | Cloud | FiQA | Govt | Total |
|--------|--------|-------|------|------|-------|
| 1-5 words | 42 (20%) | 51 (27%) | 38 (21%) | 47 (23%) | 178 (23%) |
| 6-10 words | 98 (47%) | 89 (47%) | 91 (51%) | 97 (48%) | 375 (48%) |
| 11-15 words | 52 (25%) | 38 (20%) | 41 (23%) | 43 (21%) | 174 (22%) |
| 16+ words | 16 (8%) | 10 (5%) | 10 (6%) | 14 (7%) | 50 (6%) |

### Conversation History Length

| History Turns | Queries | Percentage |
|--------------|---------|------------|
| 0 (first turn) | 102 | 13.1% |
| 1-2 | 198 | 25.5% |
| 3-5 | 255 | 32.8% |
| 6-8 | 178 | 22.9% |
| 9+ | 44 | 5.7% |

### Entity Distribution

- Queries with entities in last turn: 412 (53%)
- Queries with entities in history: 689 (89%)
- Average entities per query: 2.3
- Average entities in history: 5.7

---
