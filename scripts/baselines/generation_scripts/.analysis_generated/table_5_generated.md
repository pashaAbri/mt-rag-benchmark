# Table 5: Generation Results by Retrieval Setting

<style>
table { color: #009900; font-weight: bold; }  /* Green bold for experimental results */
</style>

**Generated from experimental results**

Generation results by retrieval setting: Reference (•), Reference+RAG (◐), and RAG (○), w/ IDK conditioned metrics. Per column, the best result is in **bold** and second best is <u>underlined</u>.

| | Ans. Acc. | | | | RLF | | | | RBllm | | | | RBalg | | |
|-------|-----------|---|---|---|-----|---|---|---|-------|---|---|---|-------|---|---|
| | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ |
| **Command-R+ (104B)** | 0.70 | 0.62 | 0.66 |  | 0.69 | 0.73 | 0.74 |  | 0.63 | 0.59 | 0.55 |  | 0.40 | 0.39 | 0.38 |
| **GPT-4o** | 0.77 | 0.73 | 0.72 |  | 0.69 | 0.66 | 0.66 |  | 0.61 | 0.58 | 0.55 |  | 0.39 | 0.36 | 0.34 |
| **GPT-4o-mini** | 0.85 | 0.79 | 0.80 |  | 0.70 | 0.72 | 0.70 |  | 0.66 | 0.64 | 0.60 |  | 0.39 | 0.36 | 0.34 |
| **Llama 3.1 405B Instruct** | 0.74 | 0.67 | 0.69 |  | 0.77 | 0.81 | 0.76 |  | 0.74 | 0.72 | 0.67 |  | 0.46 | 0.45 | 0.42 |
| **Llama 3.1 70B Instruct** | 0.71 | 0.71 | 0.69 |  | 0.67 | 0.68 | 0.68 |  | 0.57 | 0.57 | 0.53 |  | 0.39 | 0.38 | 0.36 |
| **Llama 3.1 8B Instruct** | 0.34 | 0.38 | 0.36 |  | 0.53 | 0.54 | 0.52 |  | 0.42 | 0.40 | 0.38 |  | 0.32 | 0.31 | 0.29 |
| **Mixtral 8x22B Instruct** | 0.74 | 0.66 | 0.72 |  | 0.77 | 0.81 | 0.80 |  | 0.73 | 0.70 | 0.66 |  | 0.47 | 0.44 | 0.43 |
| **Qwen 2.5 (72B)** | 0.76 | 0.71 | 0.76 |  | 0.74 | 0.75 | 0.74 |  | 0.75 | 0.74 | 0.70 |  | 0.43 | 0.42 | 0.40 |
| **Qwen 2.5 (7B)** | 0.75 | 0.69 | 0.73 |  | 0.74 | 0.76 | 0.75 |  | 0.66 | 0.65 | 0.61 |  | 0.42 | 0.40 | 0.39 |

## Legend

- **•** = Reference (perfect retrieval - reference passages only)
- **◐** = Reference+RAG (reference passages + top retrieved to reach 5 passages)
- **○** = Full RAG (top 5 retrieved passages using Elser with query rewrite)
