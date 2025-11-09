# Table 16c: Generation Results by Domain

<style>
table { color: #0066CC; }
</style>

**From MT-RAG Paper - Table 16c (Appendix I)**

Detailed generation results in the Reference (•) retrieval setting using three metrics (RLF, RBllm, RBalg) broken down by domain.

**Note:** **Bold values** indicate the best-performing model (excluding Reference) for each metric-domain combination. <u>Underlined values</u> indicate the second-best performing model.

## Results by Domain

| Model | RLF | | | | | RBllm | | | | | RBalg | | | |
|-------|---------|------|------|-------|---|---------|------|------|-------|---|---------|------|------|-------|
| | CLAPNQ | FiQA | Govt | Cloud | | CLAPNQ | FiQA | Govt | Cloud | | CLAPNQ | FiQA | Govt | Cloud |
| **Reference (Paper)** | 0.86 | 0.89 | 0.85 | 0.87 | | 0.96 | 0.95 | 0.95 | 0.95 | | 0.88 | 0.88 | 0.88 | 0.87 |
| **Command-R+ (104B)** | **0.78** | <u>0.76</u> | <u>0.76</u> | 0.73 | | 0.72 | 0.70 | 0.68 | 0.66 | | 0.47 | <u>0.39</u> | 0.43 | 0.45 |
| **GPT-4o** | <u>0.73</u> | <u>0.74</u> | **0.77** | **0.78** | | **0.77** | **0.74** | **0.78** | **0.74** | | **0.48** | 0.38 | <u>0.47</u> | 0.45 |
| **GPT-4o-mini** | 0.69 | 0.71 | 0.71 | 0.75 | | <u>0.76</u> | **0.74** | 0.75 | <u>0.73</u> | | 0.45 | 0.37 | 0.44 | 0.44 |
| **Llama 3.1 405B Inst.** | 0.72 | **0.76** | **0.77** | <u>0.76</u> | | 0.75 | <u>0.73</u> | 0.75 | 0.72 | | **0.48** | **0.41** | **0.49** | **0.49** |
| **Llama 3.1 70B Inst.** | 0.66 | 0.67 | 0.73 | 0.71 | | 0.66 | 0.62 | 0.69 | 0.65 | | 0.44 | 0.37 | 0.45 | <u>0.47</u> |
| **Llama 3.1 8B Inst.** | 0.52 | 0.56 | 0.56 | 0.56 | | 0.62 | 0.55 | 0.60 | 0.57 | | 0.37 | 0.32 | 0.38 | 0.38 |
| **Mixtral 8x22B Inst.** | 0.60 | 0.60 | 0.61 | 0.64 | | 0.72 | 0.68 | 0.70 | 0.68 | | 0.45 | 0.34 | 0.42 | 0.44 |
| **Qwen 2.5 (72B)** | 0.68 | 0.70 | 0.74 | <u>0.76</u> | | 0.75 | <u>0.73</u> | <u>0.76</u> | <u>0.73</u> | | <u>0.46</u> | 0.37 | 0.46 | 0.46 |
| **Qwen 2.5 (7B)** | 0.70 | 0.63 | 0.68 | 0.69 | | 0.72 | 0.69 | 0.74 | 0.71 | | **0.48** | 0.35 | 0.44 | 0.44 |


