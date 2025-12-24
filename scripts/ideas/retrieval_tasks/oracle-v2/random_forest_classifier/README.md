# Random Forest Classifier for Retrieval Routing

## Motivation

### The Oracle Performance Gap

Our oracle analysis revealed significant potential for improvement in retrieval performance:

- **Single-retriever oracle**: Selecting the best strategy per query achieves **+17.22% improvement** on nDCG@10
- **Cross-retriever oracle**: Selecting the best retriever+strategy combination achieves **+30.42% improvement** on nDCG@10

However, the oracle approach has critical limitations:

1. **Computational Expense**: Requires running all 9 retrieval combinations (3 retrievers × 3 strategies) for every query
2. **Requires Ground Truth**: Needs access to relevance judgments (qrels) to determine which combination is "best"
3. **Not Practical for Production**: Cannot be used in real-time systems where we need to route queries before retrieval

### The Solution: Pre-Retrieval Routing

We need a **pre-retrieval routing system** that can predict the optimal retriever+strategy combination **before** performing any retrievals, using only:

- Query characteristics (length, structure, content)
- Conversation context (turn number, history)
- Metadata enrichments (question type, answerability, multi-turn type)
- Domain information

This enables intelligent routing with **1/9th the computational cost** while capturing most of the oracle's performance gains.

## What is a Random Forest?

### Overview

A **Random Forest** is an ensemble machine learning algorithm that combines multiple decision trees to make predictions. It's particularly well-suited for classification tasks like ours.

### How It Works

1. **Multiple Trees**: Creates many decision trees (default: 100), each trained on a random subset of the data
2. **Random Feature Selection**: Each tree only considers a random subset of features when making splits
3. **Voting**: For classification, all trees vote on the predicted class, and the majority vote wins
4. **Averaging**: For regression, predictions are averaged across all trees

### Why Random Forest for Routing?

**Advantages:**
- **Handles Mixed Data Types**: Works well with both numeric (query length, turn ID) and categorical (domain, question type) features
- **Feature Importance**: Provides interpretable feature importance scores
- **Robust to Overfitting**: Less prone to overfitting than single decision trees
- **No Feature Scaling Required**: Works well with features on different scales
- **Handles Missing Values**: Can work with incomplete data
- **Fast Training**: Efficient training and prediction

**Key Hyperparameters:**
- `n_estimators`: Number of trees (more trees = better but slower)
- `max_depth`: Maximum depth of trees (controls complexity)
- `min_samples_split`: Minimum samples needed to split a node
- `min_samples_leaf`: Minimum samples required at a leaf node

### Decision Tree Example

A single decision tree might look like:

```
Is query_length_words > 10?
├─ Yes → Is turn_id > 2?
│   ├─ Yes → Predict "elser_rewrite"
│   └─ No → Predict "bm25_lastturn"
└─ No → Is domain == "clapnq"?
    ├─ Yes → Predict "bge_questions"
    └─ No → Predict "elser_lastturn"
```

A Random Forest combines hundreds of such trees, each trained on different data subsets and feature combinations, then votes on the final prediction.

## Training Process

### Step 1: Data Preparation

The training script (`train_model.py`) performs the following steps:

1. **Load Combined Data**: Reads the CSV file with features and oracle selections
2. **Feature Selection**: Excludes metadata columns (task_id, conversation_id) and target variables
3. **Categorical Encoding**: Converts categorical features (domain, question_type, etc.) to numeric using Label Encoding
4. **Missing Value Handling**: Fills missing numeric values with median, categorical with 'MISSING'
5. **Train/Test Split**: Splits data into training (80%) and test (20%) sets, maintaining class distribution

### Step 2: Model Training

The Random Forest classifier is trained with the following parameters:

- **Number of Trees**: 100 (configurable via `--n-estimators`)
- **Max Depth**: 20 (configurable via `--max-depth`)
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Random State**: 42 (for reproducibility)

### Step 3: Evaluation

During training, the model is evaluated on the test set:

- **Accuracy**: Percentage of correct predictions
- **Classification Report**: Precision, recall, F1-score per class
- **Feature Importance**: Which features are most predictive

### Step 4: Model Persistence

The trained model and artifacts are saved:

- `random_forest_model.pkl`: Trained model (can be loaded with joblib)
- `label_encoders.json`: Encoders for categorical features (needed for prediction)
- `feature_importance.json`: Feature importance rankings
- `training_results.json`: Training metrics and parameters

## Testing/Evaluation Process

### Step 1: Load Model

The evaluation script (`evaluate_model.py`) loads:

- Trained Random Forest model
- Label encoders (to transform new data the same way as training)

### Step 2: Prepare Test Data

For each task in the evaluation set:

1. Extract features (same as training)
2. Apply label encoders
3. Handle missing values

### Step 3: Make Predictions

The model predicts the optimal combination for each query.

### Step 4: Calculate Performance Metrics

Two types of metrics are calculated:

#### 1. Routing Accuracy
- **Definition**: Percentage of queries where the model predicts the same combination as the oracle
- **Interpretation**: How often the model makes the "correct" routing decision

#### 2. Performance Scores
- **Predicted Score**: Average nDCG@5 of the model's predicted combinations
- **Oracle Score**: Average nDCG@5 of the oracle's optimal combinations
- **Performance Gap**: Difference between oracle and predicted scores
- **Performance Captured**: Percentage of oracle's performance gain captured by the model

### Step 5: Generate Reports

The evaluation produces:

- Console output with key metrics
- Detailed classification report (per-class metrics)
- `evaluation_summary.json`: JSON file with all metrics

## Results

### Model Performance

Based on evaluation of the trained Random Forest classifier:

#### Routing Accuracy
- **76.8%**: The model correctly predicts the oracle's choice 76.8% of the time

#### Performance Capture
- **93.1%**: The model captures 93.1% of the oracle's performance gain
- **Average Predicted Score**: 0.5725 (nDCG@5)
- **Average Oracle Score**: 0.6147 (nDCG@5)
- **Performance Gap**: 0.0422 (small gap compared to oracle)

### Feature Importance

The most important features for routing decisions:

1. **Query Length (chars)**: 20.8% importance
   - Longer queries may benefit from different retrieval strategies

2. **Query Length (words)**: 15.4% importance
   - Word count provides complementary information to character count

3. **Turn ID**: 15.3% importance
   - Later turns in conversations may need different strategies

4. **Question Type**: 14.0% importance
   - Different question types (Factoid, Explanation, etc.) benefit from different approaches

5. **Domain**: 10.1% importance
   - Domain-specific patterns (clapnq, cloud, fiqa, govt) influence optimal routing

6. **Number of Capitalized Words**: 8.0% importance
   - May indicate proper nouns or entities

7. **Has WH-word**: 4.3% importance
   - Question structure indicator

8. **Multi-Turn Type**: 4.2% importance
   - Follow-up vs clarification questions may need different strategies

### Interpretation

**Excellent Performance Capture**: The model achieves 93.1% of the oracle's performance while requiring only 1/9th the computational cost (running 1 retrieval instead of 9).

**High Routing Accuracy**: 76.8% accuracy means the model correctly identifies the optimal combination for about 3 out of 4 queries.

**Interpretable Features**: The feature importance scores reveal that query length and conversation context (turn ID) are the strongest predictors, which aligns with intuition about multi-turn conversations.

## Usage

### Training

```bash
# Basic training with default parameters
python train_model.py

# Custom parameters
python train_model.py \
    --data-file ../feature-analysis/combined_data.csv \
    --output-dir models \
    --test-size 0.2 \
    --n-estimators 200 \
    --max-depth 25
```

### Evaluation

```bash
# Full evaluation on all tasks
python evaluate_model.py

# Sample evaluation (faster, for testing)
python evaluate_model.py --sample-size 100

# Custom paths
python evaluate_model.py \
    --data-file ../feature-analysis/combined_data.csv \
    --model-dir models \
    --results-base-dir ../../../../scripts/baselines/retrieval_scripts
```

## Files

- `train_model.py`: Script to train the Random Forest classifier
- `evaluate_model.py`: Script to evaluate model performance
- `README.md`: This documentation file
- `models/`: Directory containing trained models and results
  - `random_forest_model.pkl`: Trained model
  - `label_encoders.json`: Feature encoders
  - `feature_importance.json`: Feature importance rankings
  - `training_results.json`: Training metrics
  - `evaluation_summary.json`: Evaluation metrics

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different tree counts, depths, and other parameters
2. **Feature Engineering**: Add more features (e.g., semantic embeddings, conversation history summaries)
3. **Ensemble Methods**: Combine Random Forest with other models (e.g., gradient boosting)
4. **Production Integration**: Deploy the model as a pre-retrieval routing service
5. **A/B Testing**: Compare routed vs non-routed retrieval in production

