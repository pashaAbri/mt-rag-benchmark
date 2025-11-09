# Table 16a: Generation Results by Answerability

<style>
table { color: #0066CC; }
</style>

**From MT-RAG Paper - Table 16a (Appendix I)**

Detailed generation results in the Reference (•) retrieval setting using three metrics (RLF, RBllm, RBalg) broken down by question answerability.

**Note:** **Bold values** indicate the best-performing model (excluding Reference) for each metric-answerability combination. <u>Underlined values</u> indicate the second-best performing model.

## Results by Question Answerability

| | Overall | | | | Answerable | | | | Partial | | | | Unans. |
|-------|-----|-------|-------|---|-----|-------|-------|---|-----|-------|-------|---|-----|
| | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | |
| **Reference** | 0.87 | 0.95 | 0.88 | | 0.88 | 0.96 | 0.88 | | 0.71 | 0.88 | 0.83 | | 0.87 |
| **Command-R+ (104B)** | **0.76** | 0.69 | 0.44 | | **0.82** | 0.74 | <u>0.47</u> | | **0.59** | 0.63 | 0.36 | | 0.13 |
| **GPT-4o** | <u>0.75</u> | **0.76** | <u>0.45</u> | | **0.82** | **0.81** | **0.48** | | 0.53 | <u>0.71</u> | 0.35 | | 0.20 |
| **GPT-4o-mini** | 0.71 | <u>0.75</u> | 0.43 | | 0.77 | 0.79 | 0.44 | | 0.39 | 0.62 | 0.30 | | <u>0.34</u> |
| **Llama 3.1 405B Instruct** | <u>0.75</u> | 0.74 | **0.47** | | <u>0.81</u> | 0.79 | 0.50 | | <u>0.58</u> | 0.66 | <u>0.37</u> | | 0.20 |
| **Llama 3.1 70B Instruct** | 0.69 | 0.66 | 0.44 | | 0.74 | 0.69 | 0.45 | | 0.42 | 0.47 | 0.27 | | **0.44** |
| **Llama 3.1 8B Instruct** | 0.55 | 0.59 | 0.36 | | 0.59 | 0.62 | 0.38 | | 0.34 | 0.47 | 0.24 | | 0.33 |
| **Mixtral 8x22B Instruct** | 0.61 | 0.69 | 0.41 | | 0.68 | 0.75 | 0.45 | | 0.41 | 0.68 | 0.33 | | 0.00 |
| **Qwen 2.5 (72B)** | 0.72 | 0.74 | 0.44 | | 0.79 | <u>0.80</u> | <u>0.47</u> | | 0.53 | **0.72** | **0.38** | | 0.07 |
| **Qwen 2.5 (7B)** | 0.68 | 0.72 | 0.43 | | 0.74 | 0.77 | 0.46 | | 0.44 | 0.67 | 0.36 | | 0.11 |


