# Feature Analysis for Oracle Routing

This directory contains scripts for extracting features and analyzing patterns to predict optimal retriever+strategy combinations.

## Scripts

### 1. `extract_oracle_selections.py`

Extracts per-task oracle selections (which retriever+strategy combination is optimal for each task) based on nDCG@5.

**Usage:**
```bash
python extract_oracle_selections.py --domain all --output oracle_selections.json
```

**Output:** `oracle_selections.json` containing:
- Per-task oracle selections (best combination for each task_id)
- Combination distribution statistics

### 2. `extract_features.py`

Extracts features from task data including:
- Query characteristics (length, structure, keywords)
- Enrichments (answerability, question type, multi-turn type)
- Conversation context (turn number, domain)

**Usage:**
```bash
python extract_features.py --domain all --output task_features.json
```

**Output:** `task_features.json` containing:
- Per-task feature vectors
- Feature statistics

### 3. `analyze_patterns.py`

Analyzes patterns between features and oracle selections to identify predictive signals.

**Usage:**
```bash
python analyze_patterns.py \
    --oracle-file oracle_selections.json \
    --features-file task_features.json \
    --output-dir analysis_results
```

**Output:**
- `combined_data.csv`: Combined dataframe with features and oracle selections
- `analysis_*.json`: Analysis results for each feature
- Console output with summary statistics and patterns

## Workflow

1. **Extract oracle selections:**
   ```bash
   python extract_oracle_selections.py --domain all
   ```

2. **Extract task features:**
   ```bash
   python extract_features.py --domain all
   ```

3. **Analyze patterns:**
   ```bash
   python analyze_patterns.py
   ```
   This generates:
   - `combined_data.csv`: Combined features and oracle selections
   - `analysis_*.json`: Feature-specific analysis files
   - `bm25_vs_elser_analysis.json`: Specific analysis for BM25 vs ELSER cases
   - `lastturn_vs_rewrite_analysis.json`: Specific analysis for Lastturn vs Rewrite

4. **Train routing model:**
   ```bash
   python train_routing_model.py --data-file combined_data.csv
   ```

5. **Evaluate routing model:**
   ```bash
   python evaluate_routing.py --data-file combined_data.csv --model-dir models
   ```

## Features Extracted

### Query Characteristics
- `query_length_chars`: Character count
- `query_length_words`: Word count
- `has_question_mark`: Boolean
- `has_wh_word`: Boolean (what, who, where, etc.)
- `num_capitalized_words`: Count of capitalized words
- `num_numbers`: Count of numeric tokens

### Enrichments
- `answerability`: ANSWERABLE, UNANSWERABLE, PARTIAL, CONVERSATIONAL
- `question_type`: Factoid, Explanation, Composite, etc.
- `multi_turn_type`: Follow-up, Clarification, N/A

### Context
- `domain`: clapnq, cloud, fiqa, govt
- `turn_id`: Turn number in conversation
- `is_first_turn`: Boolean
- `conversation_id`: Conversation identifier
- `conversation_length`: Total number of user turns in conversation
- `num_previous_turns`: Number of previous turns
- `previous_turn_query_length`: Word count of previous turn query
- `previous_turn_question_type`: Question type of previous turn
- `previous_turn_multi_turn_type`: Multi-turn type of previous turn

## Oracle Selection Metric

The oracle selections are based on **nDCG@5** performance, selecting the retriever+strategy combination that maximizes nDCG@5 for each task.

## Scripts Overview

### 1. `extract_oracle_selections.py`
Extracts per-task oracle selections based on nDCG@5.

### 2. `extract_features.py`
Extracts comprehensive features including:
- Query characteristics
- Enrichments
- Conversation history (if conversations.json available)

### 3. `analyze_patterns.py`
Performs comprehensive pattern analysis:
- Summary statistics
- Categorical feature analysis
- Numeric feature analysis (with binning)
- **BM25 vs ELSER analysis**: Identifies when BM25 outperforms ELSER
- **Lastturn vs Rewrite analysis**: Identifies when each strategy is optimal

### 4. `train_routing_model.py`
Trains a Random Forest classifier to predict optimal combinations:
- Uses all extracted features
- Outputs model, feature importance, and evaluation metrics

### 5. `evaluate_routing.py`
Evaluates routing model performance:
- Calculates routing accuracy
- Measures how well routing captures oracle performance gains
- Compares predicted vs oracle scores

## Analysis Outputs

The analysis generates several files:
- `combined_data.csv`: Full dataset with features and oracle selections
- `analysis_*.json`: Feature-specific pattern analysis
- `bm25_vs_elser_analysis.json`: When BM25 beats ELSER
- `lastturn_vs_rewrite_analysis.json`: Strategy selection patterns
- `models/routing_model.pkl`: Trained routing model
- `models/feature_importance.json`: Feature importance rankings

