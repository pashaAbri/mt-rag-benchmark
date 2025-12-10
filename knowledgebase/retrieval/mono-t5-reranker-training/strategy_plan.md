# Strategy: Relevance Scoring for Query Strategy Selection

## Executive Summary
This document outlines "Approach A," a method for selecting the best query formulation strategy (`rewrite`, `lastturn`, `questions`) by using a fine-tuned MonoT5 Cross-Encoder to judge the **quality of retrieved documents**. 

Instead of trying to predict the best strategy from the query string alone (which has proven difficult), this approach runs retrieval for multiple strategies and selects the result set that contains the most relevant documents.

## 1. The Core Objective
**Goal**: Judge the quality of retrieval results to switch strategies only when evidence suggests the baseline (`rewrite`) has failed and an alternative (`lastturn`) has succeeded.

**Rationale**:
- **Baseline Dominance**: The `rewrite` strategy is the best default, winning in >80% of cases or performing adequately.
- **The "Gap"**: The ~10% performance gap between `rewrite` and the Oracle comes from specific edge cases where `rewrite` introduces noise or drift, while `lastturn` (or `questions`) retrieves a specific "Gold Nugget" document.
- **Evidence-Based**: We cannot detect these edge cases from the query text. We must look at the *retrieved documents* to see if they answer the query.

## 2. The Model: MonoT5 Cross-Encoder
We will fine-tune a `monot5-base` model to act as a **Relevance Judge**.

- **Architecture**: T5-base (220M params) fine-tuned on MS MARCO, then further fine-tuned on our domain data.
- **Input Format**: `Query: {query_text} Document: {document_text} Relevant:`
- **Output**: Probability of the token `"true"` (vs `"false"`).
- **Function**: Assigns a scalar score (0.0 to 1.0) representing the semantic relevance of a document to a query.

## 3. Training Strategy
The model must learn to distinguish "Gold" documents from "Trash" documents, regardless of which query strategy produced them.

### Data Construction
We will aggregate data from all available query strategies to create a robust training set.

- **Positive Examples**: `{query, doc}` pairs from the Qrels (Human-annotated relevant documents).
- **Negative Examples**: `{query, doc}` pairs from retrieval results that are **not** in the Qrels ("Hard Negatives").
- **Mixing**: 
    - Use `rewrite` queries + their retrieved docs.
    - Use `lastturn` queries + their retrieved docs.
    - This prevents the model from overfitting to a specific query style (e.g., full sentences vs keywords).

### Training Parameters
- **Loss Function**: Cross-Entropy Loss on the "true"/"false" logits.
- **Ratio**: 1 Positive : 4 Negatives (to emphasize precision).
- **Epochs**: 3-5 (with Early Stopping).

## 4. Inference Strategy (The "Tournament")
At runtime, we treat the query strategies as candidates in a tournament.

### Step A: Parallel Retrieval
Execute retrieval for the candidate strategies.
- **Candidate 1 (`rewrite`)**: The incumbent/baseline.
- **Candidate 2 (`lastturn`)**: The challenger (often captures specific keywords `rewrite` misses).
- *(Optional) Candidate 3 (`questions`)*: If compute allows.

### Step B: Relevance Scoring
Pass the `{query, doc}` pairs through the MonoT5 model.
- **Critical Detail**: Score the documents against the **specific query text** that generated them.
    - `rewrite` docs are scored against the `rewrite` query text.
    - `lastturn` docs are scored against the `lastturn` query text.

### Step C: Aggregation (The "Strategy Score")
Convert the list of document scores (e.g., top 10) into a single **Strategy Confidence Score**.

**Method**: **Max(Top-k Scores)** (e.g., Max of Top 3)
- **Why?** We are looking for **Peak Relevance**.
- If `rewrite` returns 10 mediocre documents (scores ~0.4), it indicates "Drift" or "Broad but shallow match."
- If `lastturn` returns 1 perfect document (score ~0.95) and 9 bad ones, it has found the answer.
- **Goal**: Select the strategy that found the "Gold Nugget."

### Step D: Decision Logic
Use a bias threshold to favor the baseline.

```python
score_rewrite = calculate_confidence(rewrite_docs)
score_lastturn = calculate_confidence(lastturn_docs)

# Only switch if challenger is significantly better
if score_lastturn > score_rewrite + THRESHOLD:
    return lastturn_results
else:
    return rewrite_results
```

## 5. Why This Approach Wins
1.  **Solves "Silent Failure"**: When `rewrite` hallucinates or drifts (e.g., expanding "Apple" to "Apple Inc." when the user meant the fruit), the retrieved documents won't semantically match the user's actual intent (as captured in the raw turn). *Correction: The scorer uses the query that generated the docs. If `rewrite` is bad, the docs are bad relative to the rewrite? No, the scorer checks if the doc answers the query. If the `rewrite` query itself is drifted, the docs might match the `rewrite` query but be wrong. High-level check: We might need to cross-check against the `lastturn` (raw) query to detect drift.*
    - *Refinement*: We should score **all** documents against a "Ground Truth" representation of intent? No, we don't have that.
    - *Refinement*: We assume the `rewrite` query is generally good. If `rewrite` gets bad docs (low scores), it means it failed to retrieve anything useful. If `lastturn` gets high scores, it succeeded.

2.  **Captures the "Gold Nugget"**: `lastturn` often wins because it preserves a specific keyword or entity ID that `rewrite` removed or diluted. MonoT5 is excellent at matching specific entities, so it will assign a high score to that single correct document.

3.  **Domain Agnostic**: A relevance scorer relies on semantic matching (QA logic), which generalizes across domains better than a classifier attempting to learn "query structural patterns" that correlate with strategy success.

## 6. Implementation Plan
1.  **`train_relevance_scorer.py`**: Script to build the dataset and fine-tune MonoT5.
2.  **`evaluate_selector.py`**: Script to run the tournament on test data and optimize the Aggregation/Threshold parameters.
3.  **`compare_approaches.py`**: Final validation against the `rewrite`-only baseline and the Oracle.
