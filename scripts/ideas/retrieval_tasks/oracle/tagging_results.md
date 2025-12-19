# Query Tagging with LLM - Results

This experiment uses Claude (Anthropic API) to classify incoming queries into question types, comparing predictions against human annotations.

## Experiment Setup

- **Model**: `claude-sonnet-4-5-20250929`
- **Total Samples**: 833
- **Domains**: clapnq, cloud, fiqa, govt
- **Evaluation Date**: 2025-12-13

### Question Types

The LLM classifies each query into 1-3 of these categories:

- Comparative
- Composite
- Explanation
- Factoid
- How-To
- Keyword
- Non-Question
- Opinion
- Summarization
- Troubleshooting

---

## Overall Performance

| Match Type                                 | Count | Rate      |
| ------------------------------------------ | ----- | --------- |
| **Exact Match** (all tags correct)         | 194   | **23.3%** |
| **Partial Match** (at least 1 tag overlap) | 395   | **47.4%** |
| **No Match** (complete mismatch)           | 244   | **29.3%** |

**~70.7%** of queries had at least some overlap with human annotations.

---

## Aggregate Metrics

| Metric            | Precision | Recall | F1        |
| ----------------- | --------- | ------ | --------- |
| **Micro-Average** | 0.463     | 0.583  | **0.516** |
| **Macro-Average** | 0.457     | 0.632  | **0.484** |

---

## Per-Class Performance (ranked by F1)

| Question Type       | F1    | Precision | Recall | Support | Predictions |
| ------------------- | ----- | --------- | ------ | ------- | ----------- |
| **Keyword**         | 0.699 | 0.655     | 0.750  | 76      | 87          |
| **Factoid**         | 0.663 | 0.542     | 0.854  | 274     | 432         |
| **Comparative**     | 0.609 | 0.467     | 0.875  | 48      | 90          |
| **How-To**          | 0.596 | 0.453     | 0.869  | 84      | 161         |
| **Opinion**         | 0.549 | 0.427     | 0.770  | 87      | 157         |
| **Composite**       | 0.488 | 0.645     | 0.392  | 51      | 31          |
| **Non-Question**    | 0.413 | 0.369     | 0.469  | 81      | 103         |
| **Explanation**     | 0.376 | 0.333     | 0.430  | 158     | 204         |
| **Troubleshooting** | 0.366 | 0.232     | 0.867  | 15      | 56          |
| **Summarization**   | 0.085 | 0.450     | 0.047  | 192     | 20          |

---

## Per-Domain Performance

| Domain     | Total | Exact Match | Partial Match | No Match |
| ---------- | ----- | ----------- | ------------- | -------- |
| **clapnq** | 223   | 29.2%       | 42.6%         | 28.3%    |
| **cloud**  | 203   | 28.1%       | 43.4%         | 28.6%    |
| **fiqa**   | 197   | 14.7%       | 57.9%         | 27.4%    |
| **govt**   | 210   | 20.5%       | 46.7%         | 32.9%    |

---

## Confusion Matrix (First Tag Only)

This matrix shows the count of predictions comparing human annotations (rows) vs LLM predictions (columns), considering only the first/primary tag from each.

| Human \ LLM   | Comp | Cmps | Expl | Fact | HowTo | Keyw | NonQ | Opin | Summ | Trbl | Total |
| ------------- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ----- |
| **Comp**      | 32   | 0    | 0    | 7    | 0     | 0    | 1    | 8    | 0    | 0    | 48    |
| **Cmps**      | 2    | 13   | 8    | 18   | 5     | 0    | 0    | 2    | 0    | 1    | 49    |
| **Expl**      | 4    | 1    | 31   | 67   | 10    | 6    | 8    | 11   | 1    | 5    | 144   |
| **Fact**      | 4    | 1    | 6    | 183  | 15    | 2    | 9    | 16   | 0    | 1    | 237   |
| **HowTo**     | 1    | 1    | 8    | 0    | 62    | 1    | 6    | 0    | 0    | 1    | 80    |
| **Keyw**      | 2    | 0    | 0    | 12   | 8     | 31   | 5    | 1    | 0    | 0    | 59    |
| **NonQ**      | 0    | 0    | 4    | 8    | 4     | 3    | 27   | 9    | 0    | 4    | 59    |
| **Opin**      | 3    | 0    | 8    | 10   | 5     | 0    | 3    | 26   | 0    | 0    | 55    |
| **Summ**      | 4    | 1    | 18   | 44   | 9     | 1    | 3    | 7    | 3    | 0    | 90    |
| **Trbl**      | 0    | 0    | 0    | 0    | 3     | 0    | 1    | 1    | 0    | 7    | 12    |
| **Total**     | 52   | 17   | 83   | 349  | 121   | 44   | 63   | 81   | 4    | 19   | 833   |

**Legend**: Comp=Comparative, Cmps=Composite, Expl=Explanation, Fact=Factoid, HowTo=How-To, Keyw=Keyword, NonQ=Non-Question, Opin=Opinion, Summ=Summarization, Trbl=Troubleshooting

### Key Takeaways from Confusion Matrix

- **Factoid dominates predictions**: 349 total predictions (42% of all), pulling from almost every category
- **Summarization → Factoid**: 44 out of 90 Summarization queries were predicted as Factoid (49%)
- **Explanation → Factoid**: 67 out of 144 Explanation queries were predicted as Factoid (47%)
- **Diagonal accuracy varies**: Factoid (77%), How-To (78%), Comparative (67%), Keyword (53%), but Summarization only 3%

---

## Key Observations

### Strengths

- **Keyword** queries are well-detected (F1=0.699)
- **Factoid** has good recall (85.4%) — the model catches most factual questions
- **Comparative** and **How-To** also perform reasonably well

### Weaknesses

1. **Summarization** performed terribly (F1=0.085, recall=4.7%)

   - The LLM predicted Summarization only 20 times vs 192 ground truth
   - Most were misclassified as Factoid

2. **Factoid over-prediction**: 432 predictions vs 274 ground truth

   - The model defaults to Factoid for ambiguous queries

3. **Explanation vs Factoid confusion**: Many "What is X?" and "Who is Y?" questions labeled as Explanation by humans were tagged as Factoid by the LLM

### Confusion Examples

| Query                                           | Human         | LLM                  |
| ----------------------------------------------- | ------------- | -------------------- |
| "who is Patel?"                                 | Summarization | Factoid              |
| "what did he do?"                               | Summarization | Explanation, Factoid |
| "How old is the moon?"                          | Explanation   | Factoid              |
| "where does the penobscot river start and end"  | Summarization | Factoid              |
| "What does indian agriculture produce include?" | Summarization | Factoid              |

---

## Files

- `tag_queries.py` - Main tagging script using Claude API
- `tag_queries_performance.py` - Evaluation script
- `tag_queries_performance.json` - Full performance metrics (JSON)
- `inspect_mistags.py` - Script to inspect misclassified queries
- `tagged_queries/` - Output directory with 787 tagged JSON files

---

## Potential Improvements

1. **Refine Summarization definition**: The boundary between Summarization and Factoid is unclear
2. **Add few-shot examples**: Include examples in the prompt to calibrate the model
3. **Merge similar categories**: Consider merging Explanation/Summarization or providing clearer distinctions
4. **Domain-specific tuning**: fiqa domain has notably lower exact match rate (14.7%)
