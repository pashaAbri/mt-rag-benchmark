# QPP-GenRE Experiment: Strategy Selection & Domain Fine-Tuning

This directory contains experiments using [QPP-GenRE](https://github.com/ChuanMeng/QPP-GenRE) to improve retrieval performance in the MT-RAG benchmark.

## Experiments

1.  **Experiment 1: QPP-Based Strategy Selection**
    *   Use pre-trained QPP-GenRE (Llama-3-8B-Instruct) to predict `nDCG@10` for Last Turn, Rewrite, and Questions strategies.
    *   Select the strategy with the highest predicted score per query.
    *   Evaluate against the best single strategy and Oracle routing.

2.  **Experiment 2: Domain-Specific Fine-Tuning**
    *   Fine-tune QPP-GenRE on domain-specific data (FIQA, GOVT, CLOUD, CLAPNQ).
    *   Evaluate if domain adaptation improves prediction accuracy and strategy selection.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Pre-trained Checkpoints & Setup**:
    Run the setup script to download the recommended Llama-3-8B-Instruct checkpoint.
    ```bash
    bash setup_qpp_genre.sh
    ```

## Directory Structure

*   `checkpoint/`: Stores model checkpoints (pre-trained and fine-tuned).
*   `datasets/`: Links to the main `cleaned_data` or specific dataset formats required by QPP-GenRE.
*   `output/`: Stores prediction results and evaluation metrics.
*   `QPP-GenRE-original-scripts/`: Contains the original scripts from the QPP-GenRE repository.

## Running Experiments

### Experiment 1
```bash
python run_strategy_selection.py
```

### Experiment 2
(Instructions to be added)
