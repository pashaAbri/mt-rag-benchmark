# MonoT5 Reranker Training

This directory contains scripts for training a MonoT5-based relevance scorer for query strategy selection.

## Overview

The goal is to fine-tune a MonoT5 model to judge document relevance, which will then be used to select the best query strategy (`rewrite`, `lastturn`, or `questions`) based on the quality of retrieved documents.

## Directory Structure

```
mono-t5-reranker-training/
├── prepare_training_data.py  # Script to create train/val/test splits
├── data_loader.py            # Utility to load text data at training time
├── train_model.py            # Training script for MonoT5 fine-tuning
├── data/                      # Generated training data (references only)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── metadata.json
├── checkpoints/               # Model checkpoints (created during training)
│   ├── checkpoint-*/
│   ├── final_model/
│   └── training_metadata.json
└── README.md
```

## Data Preparation

### Usage

```bash
python prepare_training_data.py
```

### Options

- `--domains`: Domains to process (default: `clapnq fiqa govt cloud`)
- `--strategies`: Query strategies to use (default: `rewrite lastturn questions`)
- `--retrieval-method`: Retrieval method for hard negatives (default: `elser`)
- `--negative-ratio`: Ratio of negatives to positives (default: `4`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--corpus-level`: Corpus level to use - `passage_level` or `document_level` (default: `passage_level`)

### Data Format

The script creates training examples as **references only** (not full text):

Each example contains:
- `query_id`: The query identifier (e.g., `dd6b6ffd177f2b311abe676261279d2f<::>2`)
- `document_id`: The document identifier (e.g., `822086267_6698-7277-0-579`)
- `label`: `1` for relevant, `0` for non-relevant
- `task_id`: The task identifier (extracted from query_id)
- `domain`: The domain name (e.g., `clapnq`)
- `strategy`: The query strategy (e.g., `rewrite`)

**At training time**, the actual text will be loaded from the original data sources:
- Query text: from `human/retrieval_tasks/{domain}/{domain}_{strategy}.jsonl`
- Document text: from `corpora/{corpus_level}/{domain}.jsonl` or retrieval results

The final MonoT5 format will be:
- **Input**: `Query: {query_text} Document: {document_text} Relevant:`
- **Output**: `true` (for relevant documents) or `false` (for non-relevant documents)

### Data Splitting

- **Split by Task ID**: To avoid data leakage, examples are split by `task_id` (not `query_id`). All turns from the same conversation stay in the same split.
  - **Why?** If Turn 2 of a conversation is in Train and Turn 3 is in Test, the model could learn conversation-specific context, entity names, or topic information from Train that leaks into Test.
  - **How?** We extract `task_id` from `query_id` by removing the turn number (e.g., `dd6b6ffd...<::>2` → `dd6b6ffd...`). All examples with the same `task_id` go to the same split.
- **Ratios**: 70% train, 15% validation, 15% test
- **Negative Sampling**: For each positive example, we sample `negative_ratio` hard negatives from retrieval results that are not in the Qrels.
- **Validation**: The script automatically verifies:
  - No `task_id` appears in multiple splits
  - No `query_id` appears in multiple splits  
  - No `(query_id, document_id)` pair appears in multiple splits

### Dataset Statistics

| Split | Conversations | Tasks (Turns) | Query-Doc Pairs (Examples) |
| :--- | :--- | :--- | :--- |
| **Train** (70%) | **77** | 536 | 16,762 |
| **Validation** (15%) | **16** | 118 | 3,607 |
| **Test** (15%) | **17** | 123 | 3,802 |
| **TOTAL** | **110** | **777** | **24,171** |

*Note: The "Tasks" (777) match the total number of answerable tasks in the MTRAG benchmark. The split ensures that all turns from a single conversation (e.g., all 5 turns of Conversation A) reside in the same split.*

## Loading Data at Training Time

Use `data_loader.py` to load actual text from references:

```python
from data_loader import DataLoader, load_example_references

# Load example references
train_examples = load_example_references(Path('data/train.jsonl'))

# Create data loader
loader = DataLoader(corpus_level='passage_level')

# Format examples for MonoT5
for ex in train_examples[:5]:
    formatted = loader.format_monot5_example(
        query_id=ex['query_id'],
        document_id=ex['document_id'],
        domain=ex['domain'],
        strategy=ex['strategy']
    )
    if formatted:
        print(f"Input: {formatted['input']}")
        print(f"Output: {'true' if ex['label'] == 1 else 'false'}")
```

## Training

### Usage

After preparing the data, run the training script:

```bash
python train_model.py
```

### Options

- `--data-dir`: Directory containing train.jsonl, val.jsonl, test.jsonl (default: `data`)
- `--output-dir`: Directory to save model checkpoints (default: `checkpoints`)
- `--model-name`: Base model name (default: `castorini/monot5-base-msmarco`)
- `--cache-dir`: Cache directory for model (optional)
- `--max-length`: Maximum sequence length (default: `512`)
- `--batch-size`: Training batch size (default: `8`)
- `--eval-batch-size`: Evaluation batch size (default: `16`)
- `--learning-rate`: Learning rate (default: `1e-5`)
- `--num-epochs`: Number of training epochs (default: `5`)
- `--warmup-steps`: Number of warmup steps (default: `500`)
- `--save-steps`: Save checkpoint every N steps (default: `500`)
- `--eval-steps`: Evaluate every N steps (default: `500`)
- `--logging-steps`: Log every N steps (default: `100`)
- `--early-stopping-patience`: Early stopping patience (default: `3`)
- `--seed`: Random seed (default: `42`)
- `--corpus-level`: Corpus level - `passage_level` or `document_level` (default: `passage_level`)
- `--fp16`: Use mixed precision training (flag)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: `1`)
- `--max-examples`: Limit number of examples for testing (default: `None`, use all)

### Examples

```bash
# Test with small subset (10 examples) - verify everything works
python train_model.py --max-examples 10 --num-epochs 1

# Basic training
python train_model.py

# With custom settings
python train_model.py \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --num-epochs 3 \
    --fp16 \
    --gradient-accumulation-steps 2
```

### Training Process

The script will:

1. **Load the base model**: Downloads `castorini/monot5-base-msmarco` if not cached
2. **Load data references**: Reads train/val examples from JSONL files
3. **Create datasets**: Uses `DataLoader` to fetch text on-the-fly during training
4. **Train**: Fine-tunes MonoT5 using Hugging Face `Trainer` with:
   - Cross-entropy loss on "true"/"false" tokens
   - Early stopping based on validation F1 score
   - Automatic checkpoint saving
5. **Evaluate**: Computes accuracy, precision, recall, and F1 on validation set
6. **Save**: Saves the best model to `checkpoints/final_model/` and training metadata

### Output

After training, you'll find:

- `checkpoints/final_model/`: The best model checkpoint (based on validation F1)
- `checkpoints/checkpoint-*/`: Intermediate checkpoints (last 3 kept)
- `checkpoints/training_metadata.json`: Training configuration and final metrics

### Model Format

The trained model expects:
- **Input**: `Query: {query_text} Document: {document_text} Relevant:`
- **Output**: Token probabilities for "true" and "false"

The model can be used for relevance scoring by comparing the logits for "true" vs "false" tokens.
