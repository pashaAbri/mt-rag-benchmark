# ELSER Relevance Scores Analysis

This document analyzes ELSER (Elastic Learned Sparse Encoder) relevance scores across different enrichment subtypes and query strategies. Unlike retrieval metrics (Recall, nDCG), relevance scores represent the **model's confidence** in the top-ranked document match.

## Overview

**What are ELSER Relevance Scores?**

- Unbounded positive floats derived from the dot product of sparse vector representations
- Higher scores indicate stronger semantic match between query and document
- Typical range: 0-30, with scores >20 indicating strong matches
- Scores are relative and should be compared within the same index/domain

**Query Strategies Analyzed:**

- **Last Turn**: Uses only the current question (no conversation context)
- **Rewrite**: Uses LLM-rewritten question with full conversation context
- **Questions**: Uses original self-contained questions (No Agent Response format)

---

## Question Types - Mean Relevance Scores

| Subtype         | Count | %     | Last Turn | Rewrite   | Questions | Best Strategy |
| :-------------- | :---- | :---- | :-------- | :-------- | :-------- | :------------ |
| Troubleshooting | 15    | 1.5%  | 19.72     | 20.58     | **26.19** | Questions     |
| Non-Question    | 69    | 6.9%  | 17.61     | 20.45     | **22.89** | Questions     |
| Opinion         | 80    | 8.0%  | 18.29     | 20.88     | **22.70** | Questions     |
| Keyword         | 73    | 7.3%  | 16.65     | 20.45     | **22.28** | Questions     |
| Factoid         | 255   | 25.4% | 19.03     | 21.37     | **22.10** | Questions     |
| Summarization   | 192   | 19.1% | 18.33     | 20.82     | **21.88** | Questions     |
| Explanation     | 148   | 14.7% | 19.27     | 21.30     | **21.33** | Questions     |
| Composite       | 49    | 4.9%  | 19.72     | 21.57     | **21.60** | Questions     |
| Comparative     | 44    | 4.4%  | 19.85     | **22.41** | 21.75     | Rewrite       |
| How-To          | 80    | 8.0%  | 18.74     | **21.46** | 20.83     | Rewrite       |

_Total: 1005 tasks (tasks can have multiple question types)_

### Key Insights - Question Types

1. **Distribution**: **Factoid** questions are most common (25.4%), followed by **Summarization** (19.1%) and **Explanation** (14.7%). **Troubleshooting** is least common (1.5%).

2. **Questions Strategy Dominates**: 8 out of 10 question types achieve highest scores with the Questions strategy, particularly for:

   - **Troubleshooting** (26.19, 1.5% of tasks) - largest advantage, +5.47 over Rewrite
   - **Non-Question** (22.89, 6.9% of tasks) - queries that aren't proper questions benefit from full context
   - **Keyword** (22.28, 7.3% of tasks) - short queries perform best when expanded to natural questions

3. **Rewrite Strategy Wins for Complex Queries**:

   - **Comparative** (22.41, 4.4% of tasks) - comparing entities benefits from explicit context rewriting
   - **How-To** (21.46, 8.0% of tasks) - procedural questions benefit from contextualized rewrites

4. **Lowest Scores**:
   - **Keyword** Last Turn (16.65) - confirms that short keyword queries lack sufficient semantic signal
   - **Non-Question** Last Turn (17.61) - incomplete queries need context

---

## Multi-Turn Types - Mean Relevance Scores

| Subtype       | Count | %     | Last Turn | Rewrite | Questions | Best Strategy |
| :------------ | :---- | :---- | :-------- | :------ | :-------- | :------------ |
| Follow-up     | 574   | 73.9% | 18.47     | 21.06   | **21.67** | Questions     |
| N/A           | 102   | 13.1% | 21.22     | 21.22   | **21.22** | All Equal     |
| Clarification | 101   | 13.0% | 17.28     | 21.29   | **23.15** | Questions     |

_Total: 777 tasks_

### Key Insights - Multi-Turn Types

1. **Distribution**: **Follow-up** questions dominate (73.9% of tasks), while **Clarification** and **N/A** each represent ~13% of tasks.

2. **Clarification Questions Struggle Most**:

   - Lowest Last Turn score (17.28, 13.0% of tasks) - short queries like "Why?" or "What about X?" lack context
   - Large gap between Last Turn and Questions (+5.87) - demonstrates critical need for full context
   - Even Rewrite (21.29) falls short of Questions (23.15), suggesting clarifications are hard to rewrite effectively

3. **Follow-up Questions**:

   - Most common type (73.9% of tasks) - moderate improvement from Rewrite (+2.59 over Last Turn)
   - Questions strategy provides best performance (+3.20 over Last Turn)

4. **N/A (Single-Turn)**:
   - All strategies perform identically (21.22, 13.1% of tasks) - expected since there's no conversation context to leverage

---

## Answerability Types - Mean Relevance Scores

| Subtype    | Count | %     | Last Turn | Rewrite | Questions | Best Strategy |
| :--------- | :---- | :---- | :-------- | :------ | :-------- | :------------ |
| ANSWERABLE | 709   | 91.2% | 18.90     | 21.27   | **21.85** | Questions     |
| PARTIAL    | 68    | 8.8%  | 16.39     | 19.51   | **21.30** | Questions     |

_Total: 777 tasks_

### Key Insights - Answerability Types

1. **Distribution**: **ANSWERABLE** questions represent the vast majority (91.2% of tasks), while **PARTIAL** questions are relatively rare (8.8% of tasks).

2. **PARTIAL Answerability Has Lower Scores**:

   - Consistently lower across all strategies compared to ANSWERABLE
   - Last Turn: 16.39 vs 18.90 (-2.51) - suggests partial questions are inherently ambiguous
   - Even Questions strategy (21.30) only reaches the level of ANSWERABLE's Rewrite (21.27)

3. **Questions Strategy Provides Largest Improvement for PARTIAL**:
   - +4.91 improvement over Last Turn (vs +2.95 for ANSWERABLE)
   - Suggests that partial questions benefit most from full context

---

## Summary Statistics

### Overall Patterns

1. **Strategy Performance Ranking**:

   - **Questions** > **Rewrite** > **Last Turn** (for most subtypes)
   - Average improvement: Questions over Last Turn = +3.0 to +5.0 points
   - Average improvement: Rewrite over Last Turn = +2.0 to +3.0 points

2. **Score Ranges**:

   - **Last Turn**: 16.39 - 21.22 (mean ~18.5)
   - **Rewrite**: 19.51 - 22.41 (mean ~21.0)
   - **Questions**: 20.83 - 26.19 (mean ~22.0)

3. **Subtypes with Highest Scores**:

   - Troubleshooting (Questions): 26.19
   - Comparative (Rewrite): 22.41
   - Non-Question (Questions): 22.89

4. **Subtypes with Lowest Scores**:
   - PARTIAL (Last Turn): 16.39
   - Keyword (Last Turn): 16.65
   - Clarification (Last Turn): 17.28

### Interpretation

- **Scores >20**: Strong semantic matches - model is confident in the top result
- **Scores 15-20**: Moderate matches - some ambiguity or partial relevance
- **Scores <15**: Weak matches - query may be too short, ambiguous, or lacks context

The analysis confirms that:

- **Context matters**: Rewrite and Questions strategies consistently outperform Last Turn
- **Question quality matters**: Well-formed questions (Questions strategy) generally achieve highest scores
- **Query type matters**: Keyword, Clarification, and PARTIAL answerability queries benefit most from context enrichment

---

## Data Source

Analysis generated by `scripts/discovery/analyze_relevance_scores.py` using:

- Task enrichments from `cleaned_data/tasks/`
- ELSER retrieval results from `scripts/baselines/retrieval_scripts/elser/results/`
- Detailed statistics available in `scripts/discovery/enrichment_analysis_results/relevance_scores_*.csv`
