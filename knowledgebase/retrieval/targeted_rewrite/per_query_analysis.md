# Targeted Rewrite: Per-Query Impact Analysis

## Overview

This document analyzes the **per-query impact** of targeted rewriting vs baseline rewriting using ELSER retrieval. While the aggregate metrics show improvement, this analysis reveals which cases benefit, which are hurt, and the underlying patterns.

**Source**: Analysis comparing `scripts/baselines/retrieval_scripts/elser/results/` (baseline) vs `scripts/ideas/retrieval_tasks/targeted_rewrite/retrieval_results/` (targeted)

---

## Overall Impact (nDCG@5)

| Category   | Count | %     | Avg Change |
| ---------- | ----- | ----- | ---------- |
| **HELPED** | 204   | 26.3% | +0.334     |
| **HURT**   | 156   | 20.1% | -0.342     |
| **SAME**   | 417   | 53.7% | —          |
| **Net**    | —     | —     | **+0.019** |

**Key Finding**: While 26% of cases improve, 20% are hurt. The magnitudes are similar (~0.34), resulting in a modest net positive.

---

## Domain Breakdown

| Domain     | Helped       | Hurt         | Net      |
| ---------- | ------------ | ------------ | -------- |
| **Govt**   | 62 (30.8%)   | 37 (18.4%)   | **+25**  |
| **ClapNQ** | 55 (26.4%)   | 37 (17.8%)   | **+18**  |
| **FiQA**   | 43 (23.9%)   | 35 (19.4%)   | **+8**   |
| **Cloud**  | 44 (23.4%)   | 47 (25.0%)   | **-3**   |

### Observations

- **Govt and ClapNQ** benefit most — open-domain questions with more tangential conversation history
- **Cloud is the only domain with net negative impact** — technical terminology gets incorrectly expanded by the LLM
- **FiQA** shows modest gains — financial domain may require more careful context handling

---

## Turn Distribution

| Turn    | Helped | Hurt | Net       |
| ------- | ------ | ---- | --------- |
| Turn 1  | 0      | 0    | 0         |
| Turn 2  | 44     | 18   | **+26**   |
| Turn 3  | 29     | 22   | +7        |
| Turn 4  | 31     | 25   | +6        |
| Turn 5  | 24     | 25   | -1        |
| Turn 6  | 22     | 21   | +1        |
| Turn 7  | 21     | 22   | -1        |
| Turn 8  | 21     | 14   | +7        |
| Turn 9  | 9      | 8    | +1        |
| Turn 10 | 2      | 1    | +1        |

### Observations

- **Turn 1 has no effect** — first turn queries have no history to filter
- **Turn 2 benefits most (+26 net)** — filtering the first turn's noise helps significantly
- **Early turns (2-4)** show consistent improvement
- **Middle turns (5-7)** are balanced — filtering is less predictable
- **Later turns (8+)** show modest improvement

---

## Question Type Analysis

### Helped Cases (204 total)

| Question Type  | Count | %     |
| -------------- | ----- | ----- |
| Factoid        | 68    | 33.3% |
| Explanation    | 26    | 12.7% |
| Summarization  | 25    | 12.3% |
| Keyword        | 20    | 9.8%  |
| Non-Question   | 17    | 8.3%  |
| Composite      | 15    | 7.4%  |

### Hurt Cases (156 total)

| Question Type  | Count | %     |
| -------------- | ----- | ----- |
| Explanation    | 38    | 24.4% |
| Factoid        | 32    | 20.5% |
| Summarization  | 21    | 13.5% |
| Opinion        | 16    | 10.3% |
| How-To         | 14    | 9.0%  |
| Composite      | 11    | 7.1%  |

### Key Difference

- **Factoid questions benefit most** (33% of helped vs 20% of hurt)
- **Explanation questions are more likely to be hurt** (24% of hurt vs 13% of helped)

---

## Query Characteristics

| Metric             | Helped Cases | Hurt Cases |
| ------------------ | ------------ | ---------- |
| Avg query length   | 8.4 words    | 7.8 words  |
| Has question mark  | 65.2%        | 71.8%      |
| Avg turn depth     | 4.8          | 5.1        |
| Multi-turn: Follow-up | 85.8%     | 86.5%      |

No strong distinguishing pattern in query surface characteristics.

---

## Sample Cases: When Targeted Rewrite Helps

These cases show the targeted rewrite successfully resolving vague, context-dependent queries:

| Original Query | Targeted Rewrite | nDCG@5 |
| -------------- | ---------------- | ------ |
| "What is it?" | "What is a Lite pricing plan in IBM Cloud?" | 0→1.0 |
| "Did they conquer Israel?" | "Did the Roman Empire conquer Israel?" | 0→1.0 |
| "registered business" | "How do I register a small business?" | 0→1.0 |
| "why don't you provide that?" | "Why doesn't the document provide specific example words..." | 0→1.0 |
| "A horse, a horse, my kingdom for a horse!" | "What Shakespeare play contains the famous line..." | 0→1.0 |
| "By the way, what is a secret?" | "What is a secret in the context of information security..." | 0→1.0 |
| "what are VPC clusters?" | "What are VPC clusters in IBM Cloud?" | 0→1.0 |
| "Restrictions to immigration." | "What are the restrictions to legal immigration?" | 0→1.0 |

### Success Pattern

The original queries are **too vague to retrieve effectively** — pronouns ("it", "they"), fragments ("registered business"), or literary quotes. The targeted rewrite adds the necessary context from conversation history.

---

## Sample Cases: When Targeted Rewrite Hurts

These cases show the targeted rewrite adding incorrect or overly-specific context:

| Original Query | Targeted Rewrite | nDCG@5 |
| -------------- | ---------------- | ------ |
| "How to create a cluster." | "How to create a **VPC** cluster in IBM Cloud..." | 1.0→0 |
| "those conversations are monitored?" | "Are conversations with **weak understanding metrics** monitored..." | 1.0→0 |
| "How many teams?" | "How many teams are in **Major League Soccer**?" | 1.0→0 |
| "How can I get my FICO score then?" | "How can I get my FICO credit score if **Credit Karma shows VantageScore**..." | 1.0→0 |
| "am I charged for those support costs as well?" | "Am I charged for IBM Cloud support costs in addition to the **resource usage charges (Fixed, Metered, Tiered, and Reserved)**?" | 1.0→0 |
| "what is NY and CA office hours?" | "What are the office hours for the **New York and California Department of Managed Health Care**..." | 1.0→0 |

### Failure Pattern

The original queries were **already specific enough** for retrieval. The LLM rewriter **hallucinated specifics** (VPC, Major League Soccer, Credit Karma) that narrow the query incorrectly, causing the retrieval to miss the relevant documents.

---

## Zero Case Impact

Replacing baseline rewrite with targeted rewrite in the 3-strategy ELSER configuration:

| Metric | Baseline Rewrite | Targeted Rewrite |
| ------ | ---------------- | ---------------- |
| Zero Cases | 98 (12.6%) | **85 (10.9%)** |
| Reduction | — | **-13 (-13.3%)** |

### Case Movement

| Movement | Count |
| -------- | ----- |
| Cases FIXED (no longer zero) | 22 |
| Cases STILL zero | 76 |
| Regressions (became zero) | 9 |
| **Net improvement** | **+13** |

---

## Recommendations

1. **Use targeted rewrite for Govt and ClapNQ domains** — consistent improvement
2. **Be cautious with Cloud domain** — technical terminology gets incorrectly expanded
3. **Consider query-specific selection**: 
   - Use targeted rewrite for vague/pronoun-heavy queries
   - Use baseline rewrite for already-specific technical queries
4. **Investigate LLM hallucination mitigation** — the failure mode is adding incorrect specifics

---

## Summary

| Aspect | Finding |
| ------ | ------- |
| Net Effect | **+1.9% nDCG@10** improvement |
| Help Rate | 26.3% of cases improved |
| Hurt Rate | 20.1% of cases degraded |
| Best Domain | Govt (+25 net), ClapNQ (+18 net) |
| Worst Domain | Cloud (-3 net) |
| Best Turns | Turn 2-4 (early conversation) |
| Success Mode | Resolving vague, context-dependent queries |
| Failure Mode | Adding hallucinated specifics to already-clear queries |

