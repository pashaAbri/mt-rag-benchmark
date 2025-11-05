# Table 16a: Generation Results by Answerability

<style>
table { color: #009900; font-weight: bold; }  /* Green bold for experimental results */
</style>

**Generated from experimental results**

Detailed generation results in the Reference (•) retrieval setting using three metrics (RLF, RBllm, RBalg) broken down by question answerability.

**Note:** **Bold values** indicate the best-performing model for each metric-answerability combination. <u>Underlined values</u> indicate the second-best performing model.

## Results by Question Answerability

| | Overall | | | | Answerable | | | | Partial | | | | Unans. |
|-------|-----|-------|-------|---|-----|-------|-------|---|-----|-------|-------|---|-----|
| | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | |
| **Command-R+ (104B)** | 0.69 | 0.63 | 0.40 |  | 0.76 | 0.64 | 0.44 |  | 0.63 | 0.59 | 0.38 |  | 0.03 |
| **GPT-4o** | 0.69 | 0.61 | 0.39 |  | 0.78 | 0.68 | 0.44 |  | 0.36 | 0.35 | 0.23 |  | 0.00 |
| **GPT-4o-mini** | 0.70 | 0.66 | 0.39 |  | 0.80 | 0.73 | 0.44 |  | 0.39 | 0.42 | 0.22 |  | 0.00 |
| **Llama 3.1 405B Instruct** | 0.77 | 0.74 | 0.46 |  | 0.84 | 0.76 | 0.51 |  | 0.66 | 0.67 | 0.39 |  | 0.04 |
| **Llama 3.1 70B Instruct** | 0.67 | 0.57 | 0.39 |  | 0.76 | 0.62 | 0.45 |  | 0.38 | 0.36 | 0.23 |  | 0.00 |
| **Llama 3.1 8B Instruct** | 0.53 | 0.42 | 0.32 |  | 0.59 | 0.44 | 0.36 |  | 0.36 | 0.33 | 0.24 |  | 0.00 |
| **Mixtral 8x22B Instruct** | 0.77 | 0.73 | 0.47 |  | 0.85 | 0.74 | 0.52 |  | 0.59 | 0.68 | 0.41 |  | 0.07 |
| **Qwen 2.5 (72B)** | 0.74 | 0.75 | 0.43 |  | 0.82 | 0.76 | 0.47 |  | 0.58 | 0.67 | 0.38 |  | 0.05 |
| **Qwen 2.5 (7B)** | 0.74 | 0.66 | 0.42 |  | 0.82 | 0.68 | 0.46 |  | 0.60 | 0.58 | 0.37 |  | 0.00 |
