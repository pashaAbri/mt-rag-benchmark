# Table 16b: Generation Results by Turn Position

<style>
table { color: #0066CC; }
</style>

**From MT-RAG Paper - Table 16b (Appendix I)**

Detailed generation results in the Reference (•) retrieval setting using three metrics (RLF, RBllm, RBalg) broken down by first turn vs subsequent turns.

**Note:** **Bold values** indicate the best-performing model (excluding Reference) for each metric-turn combination. <u>Underlined values</u> indicate the second-best performing model.

## Results by Turn Position

| | RLF | | RBllm | | RBalg | |
|-------|--------|----------|--------|----------|--------|----------|
| | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 |
| **Reference** | 0.89 | 0.86 | 0.97 | 0.95 | 0.89 | 0.87 |
| **Command-R+ (104B)** | 0.83 | **0.75** | 0.70 | 0.69 | 0.46 | 0.43 |
| **GPT-4o** | **0.86** | <u>0.74</u> | <u>0.78</u> | **0.76** | **0.54** | <u>0.44</u> |
| **GPT-4o-mini** | <u>0.84</u> | 0.69 | **0.79** | <u>0.74</u> | <u>0.50</u> | 0.42 |
| **Llama 3.1 405B Instruct** | 0.81 | <u>0.74</u> | 0.74 | <u>0.74</u> | <u>0.50</u> | **0.46** |
| **Llama 3.1 70B Instruct** | 0.80 | 0.68 | 0.69 | 0.66 | <u>0.50</u> | 0.42 |
| **Llama 3.1 8B Instruct** | 0.66 | 0.53 | 0.56 | 0.59 | 0.41 | 0.36 |
| **Mixtral 8x22B Instruct** | 0.82 | 0.58 | 0.73 | 0.69 | 0.47 | 0.40 |
| **Qwen 2.5 (72B)** | <u>0.84</u> | 0.70 | 0.77 | <u>0.74</u> | 0.51 | 0.43 |
| **Qwen 2.5 (7B)** | 0.82 | 0.65 | 0.71 | 0.72 | 0.48 | 0.42 |


