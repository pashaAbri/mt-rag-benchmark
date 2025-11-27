# Discovery Scripts

This directory contains scripts for exploratory data analysis and discovery of patterns in the mtRAG benchmark.

## Scripts

### `calculate_enrichment_stats.py`

Calculates statistics for enrichment subtypes across all tasks in `cleaned_data/tasks/`.

**Usage:**
```bash
python scripts/discovery/calculate_enrichment_stats.py
```

**Output:**
- Generates statistics tables showing counts and percentages for:
  - Question Types (10 categories)
  - Multi-Turn Types (3 categories)
  - Answerability Types (4 categories)
  - Domain distribution

**Results:**
- Statistics are displayed in the console
- Markdown tables are saved to `scripts/discovery/enrichment_stats.md`

---

### `analyze_enrichment_performance.py`

Analyzes retrieval performance by enrichment subtypes across different query strategies (lastturn, rewrite, questions).

**Usage:**
```bash
python scripts/discovery/analyze_enrichment_performance.py
```

**What it does:**
1. Loads enrichment data from `cleaned_data/tasks/`
2. Loads retrieval results from `scripts/baselines/retrieval_scripts/elser/results/`
3. Matches tasks with retrieval results by `task_id`
4. Calculates performance statistics for each enrichment subtype
5. Compares performance across strategies

**Output:**
- Summary tables comparing strategies for each enrichment type
- Key insights highlighting best/worst performing subtypes
- Detailed CSV files saved to `scripts/discovery/enrichment_analysis_results/`

**Results:**
- `enrichment_performance_question_types.csv` - Performance by question type
- `enrichment_performance_multi_turn.csv` - Performance by multi-turn type
- `enrichment_performance_answerability.csv` - Performance by answerability type

---

### `analyze_relevance_scores.py`

Analyzes ELSER relevance scores (top-1 score) by enrichment subtypes across different query strategies (lastturn, rewrite, questions).

**Usage:**
```bash
python scripts/discovery/analyze_relevance_scores.py
```

**What it does:**
1. Loads enrichment data from `cleaned_data/tasks/`
2. Loads ELSER retrieval results from `scripts/baselines/retrieval_scripts/elser/results/`
3. Extracts the top-1 relevance score for each task
4. Calculates statistics (mean, median, std dev, min, max, count) for each enrichment subtype
5. Compares relevance scores across strategies

**Output:**
- Summary tables comparing mean relevance scores for each enrichment type
- Detailed CSV files with full statistics saved to `scripts/discovery/enrichment_analysis_results/`

**Results:**
- `relevance_scores_question_types.csv` - Relevance scores by question type
- `relevance_scores_multi_turn.csv` - Relevance scores by multi-turn type
- `relevance_scores_answerability.csv` - Relevance scores by answerability type

**Note:** ELSER relevance scores are unbounded positive floats (typically 0-30, with >20 indicating strong matches). These scores represent the model's confidence in the top-ranked document match.

---

## Directory Structure

```
scripts/discovery/
├── README.md
├── calculate_enrichment_stats.py
├── analyze_enrichment_performance.py
├── analyze_relevance_scores.py
├── generate_results_summary.py
├── enrichment_stats.md (generated)
└── enrichment_analysis_results/ (generated)
    ├── enrichment_performance_question_types.csv
    ├── enrichment_performance_multi_turn.csv
    ├── enrichment_performance_answerability.csv
    ├── relevance_scores_question_types.csv
    ├── relevance_scores_multi_turn.csv
    └── relevance_scores_answerability.csv
```

---

