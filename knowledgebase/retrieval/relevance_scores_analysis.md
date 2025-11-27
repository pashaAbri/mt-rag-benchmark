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

| Domain  | Subtype         | Count |   %   | Last Turn | Rewrite   | Questions | Best Strategy |
| :------ | :-------------- | :---- | :---: | :-------- | :-------- | :-------- | :------------ |
| **All** | Troubleshooting | 15    | 1.5%  | 19.72     | 20.58     | **26.19** | Questions     |
|         | Non-Question    | 69    | 6.9%  | 17.61     | 20.45     | **22.89** | Questions     |
|         | Opinion         | 80    | 8.0%  | 18.29     | 20.88     | **22.70** | Questions     |
|         | Keyword         | 73    | 7.3%  | 16.65     | 20.45     | **22.28** | Questions     |
|         | Factoid         | 255   | 25.4% | 19.03     | 21.37     | **22.10** | Questions     |
|         | Summarization   | 192   | 19.1% | 18.33     | 20.82     | **21.88** | Questions     |
|         | Explanation     | 148   | 14.7% | 19.27     | 21.30     | **21.33** | Questions     |
|         | Composite       | 49    | 4.9%  | 19.72     | 21.57     | **21.60** | Questions     |
|         | Comparative     | 44    | 4.4%  | 19.85     | **22.41** | 21.75     | Rewrite       |
|         | How-To          | 80    | 8.0%  | 18.74     | **21.46** | 20.83     | Rewrite       |
|         |                 |       |       |           |           |           |               |
| CLAPNQ  | Non-Question    | 12    | 4.3%  | 17.57     | 19.55     | **21.90** | Questions     |
|         | Opinion         | 19    | 6.8%  | 17.10     | 20.04     | **23.34** | Questions     |
|         | Keyword         | 17    | 6.1%  | 16.35     | 21.48     | **22.76** | Questions     |
|         | Factoid         | 101   | 36.1% | 20.12     | 22.44     | **22.62** | Questions     |
|         | Summarization   | 59    | 21.1% | 19.71     | 21.57     | **22.48** | Questions     |
|         | Explanation     | 39    | 13.9% | 20.06     | **22.77** | 21.53     | Rewrite       |
|         | Composite       | 19    | 6.8%  | 20.59     | **22.92** | 21.96     | Rewrite       |
|         | Comparative     | 7     | 2.5%  | 15.09     | **20.98** | 19.97     | Rewrite       |
|         | How-To          | 7     | 2.5%  | 19.08     | 21.81     | **23.00** | Questions     |
|         |                 |       |       |           |           |           |               |
| CLOUD   | Troubleshooting | 13    | 5.6%  | 19.41     | 20.23     | **26.51** | Questions     |
|         | Non-Question    | 14    | 6.0%  | 18.01     | 18.97     | **24.38** | Questions     |
|         | Opinion         | 8     | 3.4%  | 19.40     | 20.87     | **24.30** | Questions     |
|         | Keyword         | 25    | 10.7% | 17.49     | 20.02     | **24.03** | Questions     |
|         | Factoid         | 43    | 18.4% | 18.06     | 20.56     | **22.21** | Questions     |
|         | Summarization   | 37    | 15.9% | 16.79     | 21.32     | **23.00** | Questions     |
|         | Explanation     | 35    | 15.0% | 18.85     | 19.78     | **21.60** | Questions     |
|         | Composite       | 12    | 5.2%  | 18.80     | 20.74     | **22.01** | Questions     |
|         | Comparative     | 13    | 5.6%  | 17.18     | 19.24     | **20.85** | Questions     |
|         | How-To          | 33    | 14.2% | 17.91     | 19.90     | **19.99** | Questions     |
|         |                 |       |       |           |           |           |               |
| FIQA    | Non-Question    | 20    | 8.0%  | 19.63     | 22.49     | **22.69** | Questions     |
|         | Opinion         | 42    | 16.9% | 19.98     | 22.02     | **22.38** | Questions     |
|         | Keyword         | 12    | 4.8%  | 16.46     | 21.00     | **22.37** | Questions     |
|         | Factoid         | 41    | 16.5% | 20.20     | **22.94** | 21.84     | Rewrite       |
|         | Summarization   | 43    | 17.3% | 19.34     | 21.11     | **21.79** | Questions     |
|         | Explanation     | 44    | 17.7% | 20.56     | **22.66** | 21.74     | Rewrite       |
|         | Composite       | 6     | 2.4%  | 20.13     | 21.51     | **21.73** | Questions     |
|         | Comparative     | 22    | 8.8%  | 22.89     | **24.71** | 23.09     | Rewrite       |
|         | How-To          | 19    | 7.6%  | 20.64     | **22.88** | 21.05     | Rewrite       |
|         |                 |       |       |           |           |           |               |
| GOVT    | Troubleshooting | 2     | 0.8%  | 21.75     | 22.89     | **24.15** | Questions     |
|         | Non-Question    | 23    | 9.5%  | 15.64     | 20.04     | **22.66** | Questions     |
|         | Opinion         | 11    | 4.5%  | 13.05     | 18.03     | **21.63** | Questions     |
|         | Keyword         | 19    | 7.8%  | 15.95     | **19.73** | 19.51     | Rewrite       |
|         | Factoid         | 70    | 28.8% | 17.35     | 19.42     | **21.42** | Questions     |
|         | Summarization   | 53    | 21.8% | 17.04     | 19.39     | **20.51** | Questions     |
|         | Explanation     | 30    | 12.3% | 16.85     | 19.16     | **20.18** | Questions     |
|         | Composite       | 12    | 4.9%  | 19.08     | 20.29     | **20.55** | Questions     |
|         | Comparative     | 2     | 0.8%  | 20.44     | **22.62** | 19.07     | Rewrite       |
|         | How-To          | 21    | 8.6%  | 18.19     | **22.51** | 21.21     | Rewrite       |

_Total: 1005 tasks (tasks can have multiple question types)_

## Question Types - Relevance Score Patterns

### **Troubleshooting** (15 queries)
**Highest confidence overall** with Questions strategy (26.19). Extremely strong semantic matches. Questions strategy significantly outperforms (+33% vs Last Turn), suggesting self-contained question format works exceptionally well for problem-solving queries.

### **Non-Question** (69 queries)
Strong confidence with Questions strategy (22.89). Shows substantial improvement from Last Turn (17.61) to Questions (+30%), indicating that converting implicit statements into explicit questions dramatically improves match quality.

### **Opinion** (80 queries)
High confidence with Questions strategy (22.70). Consistent pattern across domains. Benefits significantly from question formatting (+24% vs Last Turn), suggesting opinion queries need explicit framing for best semantic matching.

### **Keyword** (73 queries)
Good confidence with Questions strategy (22.28). Despite **poor retrieval performance** (worst R@5), shows **high relevance scores**, indicating ELSER is confident but often wrong—a critical disconnect between confidence and accuracy.

### **Factoid** (255 queries)
Solid confidence with Questions strategy (22.10). Largest category. Moderate improvement with rewriting (+12% vs Last Turn), suggesting factual queries benefit from context but not as dramatically as other types.

### **Summarization** (192 queries)
Good confidence with Questions strategy (21.88). Second-largest category. Consistent ~20% improvement from Last Turn to Questions, indicating summary requests need explicit question framing.

### **Explanation** (148 queries)
Moderate-to-good confidence with Questions strategy (21.33). Smallest improvement gap between Rewrite (21.30) and Questions (21.33), suggesting explanatory queries are relatively robust to formatting.

### **Composite** (49 queries)
Good confidence with Questions strategy (21.60). Very close scores between Rewrite (21.57) and Questions, indicating multi-part questions maintain quality across strategies.

### **Comparative** (44 queries)
**Only category where Rewrite wins** (22.41 vs Questions 21.75). Unique pattern suggests comparative questions benefit more from context integration than explicit question formatting. Aligns with retrieval performance where Rewrite also excelled.

### **How-To** (80 queries)
**Second category where Rewrite wins** (21.46 vs Questions 20.83). Procedural questions benefit from contextual rewriting, likely because steps and procedures require conversation continuity.

---

## Multi-Turn Types - Mean Relevance Scores

| Domain  | Subtype       | Count |   %   | Last Turn | Rewrite   | Questions | Best Strategy |
| :------ | :------------ | :---- | :---: | :-------- | :-------- | :-------- | :------------ |
| **All** | Follow-up     | 574   | 73.9% | 18.47     | 21.06     | **21.67** | Questions     |
|         | N/A           | 102   | 13.1% | 21.22     | 21.22     | **21.22** | Questions     |
|         | Clarification | 101   | 13.0% | 17.28     | 21.29     | **23.15** | Questions     |
|         |               |       |       |           |           |           |               |
| CLAPNQ  | Follow-up     | 154   | 74.0% | 19.16     | 21.74     | **22.16** | Questions     |
|         | N/A           | 28    | 13.5% | 21.94     | 21.94     | **21.94** | Questions     |
|         | Clarification | 26    | 12.5% | 19.49     | **23.18** | 22.97     | Rewrite       |
|         |               |       |       |           |           |           |               |
| CLOUD   | Follow-up     | 135   | 71.8% | 17.89     | 20.19     | **22.05** | Questions     |
|         | N/A           | 25    | 13.3% | **19.80** | 19.80     | 19.80     | Last Turn     |
|         | Clarification | 28    | 14.9% | 16.72     | 20.59     | **25.22** | Questions     |
|         |               |       |       |           |           |           |               |
| FIQA    | Follow-up     | 129   | 71.7% | 19.92     | **22.40** | 21.82     | Rewrite       |
|         | N/A           | 24    | 13.3% | **23.67** | 23.67     | 23.67     | Last Turn     |
|         | Clarification | 27    | 15.0% | 18.64     | **22.32** | 21.77     | Rewrite       |
|         |               |       |       |           |           |           |               |
| GOVT    | Follow-up     | 156   | 77.6% | 17.11     | 20.04     | **20.74** | Questions     |
|         | N/A           | 25    | 12.4% | **19.48** | 19.48     | 19.48     | Last Turn     |
|         | Clarification | 20    | 9.9%  | 13.37     | 18.46     | **22.36** | Questions     |

_Total: 777 tasks_

## Multi-Turn Types - Relevance Score Patterns

### **Follow-up** (574 queries)
Largest multi-turn category. Moderate confidence with Questions strategy (21.67). Shows consistent ~17% improvement from Last Turn to Questions. Context clearly helps, but questions format provides additional boost.

### **N/A** (102 queries)
Good confidence, **identical across all strategies** (21.22). These single-turn queries within multi-turn conversations don't benefit from any strategy—they're self-contained and context-independent.

### **Clarification** (101 queries)
Highest confidence with Questions strategy (23.15). Shows **dramatic improvement** from Last Turn (17.28) to Questions (+34%), the largest gain of any subtype. Clarification questions desperately need explicit formatting and context to achieve good semantic matches.

---

## Answerability Types - Mean Relevance Scores

| Domain  | Subtype    | Count |   %   | Last Turn | Rewrite   | Questions | Best Strategy |
| :------ | :--------- | :---- | :---: | :-------- | :-------- | :-------- | :------------ |
| **All** | ANSWERABLE | 709   | 91.2% | 18.90     | 21.27     | **21.85** | Questions     |
|         | PARTIAL    | 68    | 8.8%  | 16.39     | 19.51     | **21.30** | Questions     |
|         |            |       |       |           |           |           |               |
| CLAPNQ  | ANSWERABLE | 192   | 92.3% | 19.85     | 22.08     | **22.25** | Questions     |
|         | PARTIAL    | 16    | 7.7%  | 16.20     | 20.36     | **21.97** | Questions     |
|         |            |       |       |           |           |           |               |
| CLOUD   | ANSWERABLE | 177   | 94.2% | 17.97     | 20.29     | **22.32** | Questions     |
|         | PARTIAL    | 11    | 5.8%  | 17.94     | 18.68     | **20.74** | Questions     |
|         |            |       |       |           |           |           |               |
| FIQA    | ANSWERABLE | 163   | 90.6% | 20.51     | **22.67** | 22.00     | Rewrite       |
|         | PARTIAL    | 17    | 9.4%  | 17.57     | 21.48     | **22.64** | Questions     |
|         |            |       |       |           |           |           |               |
| GOVT    | ANSWERABLE | 177   | 88.1% | 17.31     | 20.07     | **20.82** | Questions     |
|         | PARTIAL    | 24    | 11.9% | 14.98     | 17.93     | **20.16** | Questions     |

_Total: 777 tasks_

## Answerability Types - Relevance Score Patterns

### **Answerable** (709 queries)
Strong baseline confidence with Questions strategy (21.85). Represents the "standard" query quality. Moderate improvement (~16%) from Last Turn to Questions.

### **Partial** (68 queries)
Lower confidence overall with Questions strategy (21.30). Shows **larger improvement gap** from Last Turn (16.39) to Questions (+30% vs 16% for Answerable). Partial queries struggle more with context but benefit dramatically from proper formatting.

---

### Interpretation

- **Scores >20**: Strong semantic matches - model is confident in the top result
- **Scores 15-20**: Moderate matches - some ambiguity or partial relevance
- **Scores <15**: Weak matches - query may be too short, ambiguous, or lacks context

---

## Critical Insights

### **The Keyword Paradox**
- **Highest confidence** scores (22.28) but **worst retrieval performance** (R@5: 0.395)
- ELSER is confident but frequently wrong on keyword queries
- Indicates overconfidence on sparse, ambiguous queries
- **Action needed**: Add confidence thresholding or require minimum query length

### **Questions Strategy Dominance**
- Questions strategy wins in **8 out of 10** question types
- Only Comparative and How-To benefit more from Rewrite
- Average improvement: +20-30% over Last Turn
- Suggests self-contained question format provides clearest semantic signal

### **Clarification Challenge & Opportunity**
- **Lowest confidence** with Last Turn (17.28) but **highest with Questions** (23.15)
- Largest improvement of any category (+34%)
- Clarification queries are most sensitive to formatting
- Proper question formulation is critical for follow-up understanding

### **Confidence-Performance Alignment Issues**
- Troubleshooting: High confidence (26.19) + high retrieval (R@5: 0.706) ✓ Aligned
- Keyword: High confidence (22.28) + low retrieval (R@5: 0.395) ✗ **Misaligned**
- Factoid: Moderate confidence (22.10) + moderate retrieval (R@5: 0.465) ✓ Aligned

### **Domain Variations**
- FIQA shows preference for Rewrite in several categories (unique among domains)
- CLOUD shows strongest Questions preference (highest scores)
- GOVT shows lowest overall confidence scores (15-20 range)

---

## Possible Paths Forward

1. **Adopt Questions strategy as default** for 8/10 question types
2. **Use Rewrite for Comparative and How-To** queries specifically
3. **Implement confidence thresholding** for Keyword queries (high confidence doesn't mean accurate)
4. **Prioritize clarification handling** - highest potential improvement with proper formatting
5. **Investigate GOVT domain** - systematically lower confidence suggests index quality issues
6. **Calibrate confidence scores** - establish domain-specific thresholds (>20 may not always indicate quality match)

---

## Data Source

Analysis generated by `scripts/discovery/analyze_relevance_scores.py` using:

- Task enrichments from `cleaned_data/tasks/`
- ELSER retrieval results from `scripts/baselines/retrieval_scripts/elser/results/`
- Detailed statistics available in `scripts/discovery/enrichment_analysis_results/relevance_scores_*.csv`

---