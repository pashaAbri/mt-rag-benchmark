# Oracle Query Routing Analysis

This directory contains implementations for query-based strategy routing in multi-turn conversational RAG.

## Goal

Determine if a query needs conversation context (→ use `rewrite` strategy) or is self-contained (→ use `lastturn` strategy).

## Directory Structure

```
oracle/
├── rule_based_tagger.py     # Fast regex-based tagging
├── llm_based_tagger.py      # LLM-powered tagging (Sonnet 4.5)
├── evaluate_taggers.py      # Compare taggers vs oracle
├── evaluation_results.json  # Evaluation metrics
├── README.md                # This file
└── tagged_queries/          # Output: tagged task files
    ├── clapnq/              # 224 files
    ├── cloud/               # 205 files
    ├── fiqa/                # 199 files
    └── govt/                # 214 files
```

## Implementations

### 1. Rule-Based Tagger (`rule_based_tagger.py`)

Fast, deterministic tagging using regex patterns derived from analysis:

- **Pronouns**: `it`, `this`, `that`, `they` → likely needs context
- **Fragments**: ≤4 words or ≤25 chars → likely needs context
- **Implicit references**: "the song", "the movie" → needs context
- **Anaphoric expressions**: "what about", "how about" → needs context

```bash
python rule_based_tagger.py
```

**Input**: `cleaned_data/tasks/{domain}/*.json`  
**Output**: `tagged_queries/{domain}/*.json` (with `oracle_metadata.rule_based_tags`)

### 2. LLM-Based Tagger (`llm_based_tagger.py`)

Uses Claude Sonnet 4.5 to analyze context dependency with reasoning.

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key

# Run tagger (processes all 842 queries)
python llm_based_tagger.py --workers 8
```

**Input**: `cleaned_data/tasks/{domain}/*.json` (or already-tagged files)  
**Output**: `tagged_queries/{domain}/*.json` (adds `oracle_metadata.llm_based_tags`)

### 3. Evaluation (`evaluate_taggers.py`)

Compare both taggers against oracle data (which strategy actually works best).

```bash
python evaluate_taggers.py
```

**Output**: `evaluation_results.json`

## Data Flow

1. **Source**: `cleaned_data/tasks/{domain}/*.json` - Original task files
2. **Tagging**: Both taggers add their tags to `oracle_metadata` in the output files
3. **Evaluation**: Compare predictions against `routing_analysis_results/oracle_best_strategy.csv`

## Sample Output Structure

Each tagged file in `tagged_queries/` contains the original task data plus:

```json
{
  "task_id": "abc123<::>1",
  "user": { "text": "What about the other one?" },
  "oracle_metadata": {
    "rule_based_tags": {
      "tagger": "rule_based_v1",
      "needs_context": true,
      "confidence": 0.85,
      "recommended_strategy": "rewrite",
      "is_fragment": true,
      "has_pronoun": false,
      "has_demonstrative": true,
      "matched_patterns": ["fragment", "demonstrative:other"]
    },
    "llm_based_tags": {
      "tagger": "llm_sonnet_4.5",
      "model": "claude-sonnet-4-5-20250929",
      "needs_context": true,
      "confidence": 0.95,
      "recommended_strategy": "rewrite",
      "has_unresolved_pronouns": false,
      "has_implicit_references": true,
      "has_anaphoric_expressions": true,
      "is_incomplete_fragment": true,
      "reasoning": "Query contains 'the other one' which implicitly references something from prior context."
    }
  }
}
```

## Results Summary

| Metric | Rule-Based | LLM-Based (Sonnet 4.5) | Oracle |
|--------|------------|------------------------|--------|
| Accuracy | 52.1% | TBD | 100% |
| R@5 | 0.4695 | TBD | 0.5633 |
| vs Baseline | -1.4% | TBD | +18.3% |
| Gap Captured | -7.6% | TBD | 100% |

*Note: Rule-based performs worse than baseline because it's too aggressive in routing to lastturn*

## Quick Start

```bash
# Activate venv
source .venv/bin/activate

# 1. Run rule-based tagger (fast, no API needed)
python rule_based_tagger.py

# 2. Run LLM tagger (requires ANTHROPIC_API_KEY)
python llm_based_tagger.py --workers 8

# 3. Evaluate both
python evaluate_taggers.py
```
