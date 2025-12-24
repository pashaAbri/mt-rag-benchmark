# New Query Strategies Analysis

This directory contains scripts to analyze cases where existing query strategies (lastturn, rewrite, questions) perform poorly, with the goal of identifying opportunities for new query strategies.

## Scripts

### `analyze_poor_performance_cases.py`

Analyzes queries/cases where all three strategies (lastturn, rewrite, questions) perform poorly across retrievers. This analysis helps identify patterns and characteristics that could benefit from new query strategies.

#### Usage

```bash
# Analyze poor-performing cases for ELSER retriever (default)
python analyze_poor_performance_cases.py

# Analyze for a specific retriever
python analyze_poor_performance_cases.py --retriever bm25

# Analyze across all retrievers
python analyze_poor_performance_cases.py --retriever all

# Analyze for a specific domain
python analyze_poor_performance_cases.py --domain clapnq

# Adjust the threshold percentile (default: 0.25 = bottom 25%)
python analyze_poor_performance_cases.py --threshold-percentile 0.20

# Custom paths
python analyze_poor_performance_cases.py \
    --results-dir scripts/baselines/retrieval_scripts \
    --tasks-dir cleaned_data/tasks \
    --conversations-file human/conversations/conversations.json
```

#### Output

The script generates several output files:

1. **`poor_performance_analysis.txt`** - Human-readable analysis report with:
   - Summary statistics
   - Domain and turn distributions
   - Query characteristics (length, question types, etc.)
   - Sample worst-performing queries
   - Best retriever-strategy combinations for these cases

2. **`poor_performance_cases.csv`** - Detailed CSV with all features and metrics for each poor-performing case

3. **`poor_performance_summary.json`** - JSON summary with key statistics and distributions

#### What to Look For

The analysis helps identify:

- **Query patterns** that consistently fail across all strategies
- **Conversation characteristics** (turn number, conversation length) that correlate with poor performance
- **Question types** or **multi-turn types** that are particularly challenging
- **Domain-specific** issues that might need domain-specific strategies
- **Gaps** where no existing strategy works well, suggesting opportunities for new approaches

#### Example Insights

After running the analysis, you might discover:

- Queries requiring specific context extraction methods
- Multi-turn conversations needing better history summarization
- Questions that benefit from query expansion or reformulation
- Cases where combining multiple query representations could help

These insights can guide the development of new query strategies such as:
- Context-aware query expansion
- Selective history inclusion
- Domain-specific query transformations
- Multi-query fusion approaches

