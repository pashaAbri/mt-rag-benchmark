# Table 16b: Generation Results by Turn Position

<style>
table { color: #009900; font-weight: bold; }  /* Green bold for experimental results */
</style>

**Generated from experimental results**

Detailed generation results in the Reference (•) retrieval setting using three metrics (RLF, RBllm, RBalg) broken down by first turn vs subsequent turns.

**Note:** **Bold values** indicate the best-performing model for each metric-turn combination. <u>Underlined values</u> indicate the second-best performing model.

## Results by Turn Position

| | RLF | | RBllm | | RBalg | |
|-------|--------|----------|--------|----------|--------|----------|
| | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 |
| **Command-R+ (104B)** | 0.82 | 0.67 | 0.69 | 0.62 | 0.47 | 0.39 |
| **GPT-4o** | 0.73 | 0.68 | 0.63 | 0.61 | 0.44 | 0.38 |
| **GPT-4o-mini** | 0.74 | 0.70 | 0.69 | 0.66 | 0.44 | 0.38 |
| **Llama 3.1 405B Instruct** | 0.80 | 0.76 | 0.73 | 0.74 | 0.49 | 0.45 |
| **Llama 3.1 70B Instruct** | 0.74 | 0.67 | 0.61 | 0.56 | 0.45 | 0.39 |
| **Llama 3.1 8B Instruct** | 0.71 | 0.50 | 0.59 | 0.40 | 0.43 | 0.31 |
| **Mixtral 8x22B Instruct** | 0.84 | 0.75 | 0.75 | 0.73 | 0.51 | 0.46 |
| **Qwen 2.5 (72B)** | 0.81 | 0.73 | 0.73 | 0.75 | 0.47 | 0.42 |
| **Qwen 2.5 (7B)** | 0.80 | 0.73 | 0.65 | 0.67 | 0.46 | 0.41 |
