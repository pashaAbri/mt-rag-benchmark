# Table 5: Generation Results by Retrieval Setting

<style>
table { color: #0066CC; }
</style>

**From MT-RAG Paper - Table 5**

Generation results by retrieval setting: Reference (•), Reference+RAG (◐), and RAG (○), w/ IDK conditioned metrics (n = 426). Per column, the best result is in **bold** and second best is <u>underlined</u>.

| | Ans. Acc. | | | | RLF | | | | RBllm | | | | RBalg | | |
|-------|-----------|---|---|---|-----|---|---|---|-------|---|---|---|-------|---|---|
| | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ |
| **Reference** | 0.98 | 0.97 | 0.98 | | 0.86 | 0.87 | 0.67 | | 0.94 | 0.94 | 0.94 | | 0.88 | 0.88 | 0.86 |
| **Command-R+ (104B)** | 0.86 | <u>0.86</u> | **0.87** | | **0.69** | **0.71** | **0.66** | | 0.66 | 0.62 | 0.59 | | 0.43 | <u>0.40</u> | <u>0.38</u> |
| **GPT-4o** | **0.89** | <u>0.86</u> | <u>0.86</u> | | **0.69** | 0.69 | <u>0.65</u> | | **0.73** | <u>0.68</u> | **0.66** | | <u>0.46</u> | <u>0.40</u> | <u>0.38</u> |
| **GPT-4o-mini** | 0.87 | <u>0.86</u> | 0.84 | | <u>0.66</u> | 0.69 | 0.64 | | <u>0.72</u> | <u>0.68</u> | 0.64 | | 0.43 | <u>0.40</u> | 0.37 |
| **Llama 3.1 405B Instruct** | 0.87 | <u>0.86</u> | 0.85 | | **0.69** | 0.70 | 0.65 | | 0.70 | 0.68 | 0.63 | | **0.47** | **0.42** | **0.39** |
| **Llama 3.1 70B Instruct** | 0.78 | 0.83 | 0.81 | | 0.63 | 0.66 | 0.64 | | 0.62 | 0.64 | 0.59 | | 0.43 | **0.42** | **0.39** |
| **Llama 3.1 8B Instruct** | 0.71 | 0.75 | 0.74 | | 0.50 | 0.51 | 0.53 | | 0.54 | 0.56 | 0.54 | | 0.36 | 0.33 | 0.34 |
| **Mixtral 8x22B Instruct** | 0.86 | **0.87** | <u>0.86</u> | | 0.54 | 0.61 | 0.56 | | 0.66 | 0.64 | 0.61 | | 0.39 | 0.38 | 0.35 |
| **Qwen 2.5 (72B)** | 0.87 | **0.87** | **0.87** | | 0.65 | **0.71** | 0.64 | | 0.71 | **0.69** | <u>0.65</u> | | 0.43 | <u>0.40</u> | 0.37 |
| **Qwen 2.5 (7B)** | <u>0.88</u> | <u>0.86</u> | **0.87** | | 0.62 | 0.66 | 0.62 | | 0.68 | 0.65 | 0.63 | | 0.42 | 0.38 | 0.37 |

## Legend

- **•** = Reference (perfect retrieval - reference passages only)
- **◐** = Reference+RAG (reference passages + top retrieved to reach 5 passages)
- **○** = Full RAG (top 5 retrieved passages using Elser with query rewrite)

## Metrics

- **Ans. Acc.**: Answerability Accuracy - correctly identifying answerable vs unanswerable questions
- **RLF**: Reference-Less Faithfulness (RAGAS) - LLM-based faithfulness score
- **RBllm**: Reference-Based LLM judge - comprehensive quality evaluation (adapted RAD-Bench)
- **RBalg**: Reference-Based Algorithmic - harmonic mean of BertScore-Recall, BertScore-K-Precision, and ROUGE-L

## Notes

- n = 426 tasks (subset with ≤2 reference passages)
- All metrics are IDK-conditioned (reward declining unanswerable questions)
- Results show performance degradation: Reference (•) > Reference+RAG (◆) > Full RAG (◦)
- Even frontier models (GPT-4o, Llama 3.1 405B) struggle compared to reference answers

