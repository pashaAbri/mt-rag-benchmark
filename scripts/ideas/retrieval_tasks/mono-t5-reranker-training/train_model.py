#!/usr/bin/env python3
"""
Train MonoT5 relevance scorer for query strategy selection.

This script:
1. Loads the MonoT5 base model (castorini/monot5-base-msmarco)
2. Loads training/validation data using DataLoader
3. Fine-tunes the model using Hugging Face Trainer
4. Evaluates on validation set
5. Saves model checkpoints
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Import our data loader
from data_loader import DataLoader, load_example_references

# Model configuration
MONO_T5_MODEL = "castorini/monot5-base-msmarco"
OUTPUT_TRUE = "true"
OUTPUT_FALSE = "false"


class MonoT5Dataset(Dataset):
    """Dataset for MonoT5 training that loads text on-the-fly."""
    
    def __init__(self, examples: List[Dict], data_loader: DataLoader, tokenizer: T5Tokenizer, max_length: int = 512):
        """
        Args:
            examples: List of example references (from JSONL)
            data_loader: DataLoader instance to fetch text
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Load actual text
        formatted = self.data_loader.format_monot5_example(
            query_id=ex['query_id'],
            document_id=ex['document_id'],
            domain=ex['domain'],
            strategy=ex['strategy']
        )
        
        if not formatted:
            # This shouldn't happen if data prep was correct, but handle gracefully
            print(f"WARNING: Could not load text for query_id={ex['query_id']}, document_id={ex['document_id']}")
            input_text = "Query: Document: Relevant:"
            output_text = OUTPUT_FALSE
        else:
            input_text = formatted['input']
            output_text = OUTPUT_TRUE if ex['label'] == 1 else OUTPUT_FALSE
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize output (target)
        target_encoding = self.tokenizer(
            output_text,
            max_length=10,  # "true" or "false" is short
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels: replace padding token ID with -100 (ignored in loss)
        labels = target_encoding['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def create_compute_metrics(tokenizer: T5Tokenizer):
    """Create a compute_metrics function with tokenizer closure."""
    # Pre-compute token IDs
    true_token_id = tokenizer.encode(OUTPUT_TRUE, add_special_tokens=False)[0]
    false_token_id = tokenizer.encode(OUTPUT_FALSE, add_special_tokens=False)[0]
    pad_token_id = tokenizer.pad_token_id
    
    def compute_metrics(eval_pred):
        """Compute accuracy and F1 for evaluation."""
        predictions, labels = eval_pred
        
        try:
            # Predictions are generated token IDs (since predict_with_generate=True)
            # Labels are also token IDs
            
            # Helper to extract first token
            def get_first_token(seq):
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()
                elif not isinstance(seq, list):
                    seq = [seq]
                    
                for token_id in seq:
                    # Ignore padding and special tokens like start/end tokens
                    if token_id != pad_token_id and token_id != -100:
                        return int(token_id)
                return false_token_id  # Default
            
            # Process predictions
            if isinstance(predictions, (list, tuple)):
                pred_tokens = [get_first_token(p) for p in predictions]
            else:
                # Numpy array
                if len(predictions.shape) == 2:
                    # [batch, seq_len]
                    pred_tokens = [get_first_token(p) for p in predictions]
                else:
                    # [batch] (unlikely for generation)
                    pred_tokens = [int(p) if p != pad_token_id else false_token_id for p in predictions]

            # Process labels
            if isinstance(labels, (list, tuple)):
                label_tokens = [get_first_token(l) for l in labels]
            else:
                if len(labels.shape) == 2:
                    label_tokens = [get_first_token(l) for l in labels]
                else:
                    label_tokens = [int(l) if l != pad_token_id else false_token_id for l in labels]

            # Convert to binary
            y_pred = [1 if t == true_token_id else 0 for t in pred_tokens]
            y_true = [1 if t == true_token_id else 0 for t in label_tokens]
            
            # Ensure same length
            min_len = min(len(y_pred), len(y_true))
            y_pred = np.array(y_pred[:min_len])
            y_true = np.array(y_true[:min_len])
            
            if len(y_pred) == 0:
                 return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        except Exception as e:
            print(f"Warning: compute_metrics failed: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MonoT5 relevance scorer')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing train.jsonl, val.jsonl, test.jsonl')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--model-name', type=str, default=MONO_T5_MODEL,
                        help='Base model name')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for model (optional)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                        help='Evaluation batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--save-total-limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep (default: 3)')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--logging-steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--corpus-level', type=str, default='passage_level',
                        choices=['passage_level', 'document_level'],
                        help='Corpus level to use')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Limit number of examples for testing (default: None, use all)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MonoT5 Relevance Scorer Training")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    print("✓ Model loaded")
    
    # Load data references
    print("\nLoading data references...")
    train_examples = load_example_references(data_dir / 'train.jsonl')
    val_examples = load_example_references(data_dir / 'val.jsonl')
    
    # Limit examples if --max-examples is set (for testing)
    if args.max_examples is not None:
        print(f"\n⚠️  TEST MODE: Limiting to {args.max_examples} examples")
        train_examples = train_examples[:args.max_examples]
        val_examples = val_examples[:min(args.max_examples, len(val_examples))]
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")
    
    # Count positives and negatives
    train_positives = sum(1 for ex in train_examples if ex['label'] == 1)
    train_negatives = sum(1 for ex in train_examples if ex['label'] == 0)
    val_positives = sum(1 for ex in val_examples if ex['label'] == 1)
    val_negatives = sum(1 for ex in val_examples if ex['label'] == 0)
    
    print(f"  Train: {train_positives} positives, {train_negatives} negatives")
    print(f"  Val: {val_positives} positives, {val_negatives} negatives")
    
    # Create data loader
    print("\nInitializing DataLoader...")
    data_loader = DataLoader(corpus_level=args.corpus_level)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MonoT5Dataset(train_examples, data_loader, tokenizer, max_length=args.max_length)
    val_dataset = MonoT5Dataset(val_examples, data_loader, tokenizer, max_length=args.max_length)
    print("✓ Datasets created")
    
    # Calculate total steps
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.num_epochs
    
    # Adjust steps for small datasets (test mode)
    if args.max_examples is not None and len(train_dataset) < 100:
        # For small test runs, evaluate and save more frequently
        adjusted_logging_steps = max(1, min(args.logging_steps, steps_per_epoch))
        adjusted_eval_steps = max(1, min(args.eval_steps, steps_per_epoch))
        adjusted_save_steps = max(1, min(args.save_steps, steps_per_epoch))
        adjusted_warmup_steps = min(args.warmup_steps, max(1, total_steps // 4))
        print(f"\n⚠️  Adjusting steps for small dataset:")
        print(f"    logging_steps: {args.logging_steps} → {adjusted_logging_steps}")
        print(f"    eval_steps: {args.eval_steps} → {adjusted_eval_steps}")
        print(f"    save_steps: {args.save_steps} → {adjusted_save_steps}")
        print(f"    warmup_steps: {args.warmup_steps} → {adjusted_warmup_steps}")
    else:
        adjusted_logging_steps = args.logging_steps
        adjusted_eval_steps = args.eval_steps
        adjusted_save_steps = args.save_steps
        adjusted_warmup_steps = args.warmup_steps
    
    print("\nTraining configuration:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {adjusted_warmup_steps}")
    
        # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=adjusted_warmup_steps,
        logging_steps=adjusted_logging_steps,
        eval_steps=adjusted_eval_steps,
        save_steps=adjusted_save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard unless needed
        save_total_limit=args.save_total_limit,  # Keep specified number of checkpoints
        remove_unused_columns=False,
        predict_with_generate=True,  # Generate tokens for evaluation
    )
    
    # Create compute_metrics function with tokenizer
    compute_metrics_fn = create_compute_metrics(tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Create trainer
    print("\nInitializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Final evaluation on validation set...")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    print("\nValidation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save final model
    final_model_dir = output_dir / "final_model"
    print(f"\nSaving final model to {final_model_dir}...")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Save training metadata
    metadata = {
        'model_name': args.model_name,
        'training_args': vars(args),
        'final_eval_results': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                              for k, v in eval_results.items()},
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'train_positives': train_positives,
        'train_negatives': train_negatives,
        'val_positives': val_positives,
        'val_negatives': val_negatives,
    }
    
    with open(output_dir / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Training complete!")
    print(f"  Final model: {final_model_dir}")
    print(f"  Training metadata: {output_dir / 'training_metadata.json'}")


if __name__ == '__main__':
    main()
