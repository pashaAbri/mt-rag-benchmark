#!/usr/bin/env python3
"""
Run Off-the-Shelf MonoT5 (Base Model) for Query Strategy Selection on the ENTIRE Dataset.
Saves results to a single JSON file.
"""

import sys
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from utils import (
    load_retrieval_results_with_texts,
    load_queries,
    DOMAINS,
    QUERY_STRATEGIES,
)

# Configuration
MODEL_NAME = "castorini/monot5-base-msmarco"
CACHE_DIR = script_dir / ".cache"
OUTPUT_FILE = script_dir / "results" / "base_model_full_dataset_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class MonoT5Scorer:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"Loading model: {model_name} on {device}")
        
        # Use local cache if it exists
        cache_dir_str = str(CACHE_DIR) if CACHE_DIR.exists() else None
        if cache_dir_str:
            print(f"Using cache directory: {cache_dir_str}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir_str)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir_str)
        self.model.to(device)
        self.model.eval()

    def score_batch(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
            
        inputs = [f"Query: {query} Document: {doc} Relevant:" for doc in documents]
        
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        # Get scores for "true" and "false"
        true_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        false_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
        
        batch_scores = []
        for scores in outputs.scores[0]:
            true_score = scores[true_id].item()
            false_score = scores[false_id].item()
            
            # Softmax
            true_prob = torch.exp(torch.tensor(true_score))
            false_prob = torch.exp(torch.tensor(false_score))
            prob = true_prob / (true_prob + false_prob)
            batch_scores.append(prob.item())
            
        return batch_scores

def extract_task_id(query_id: str) -> str:
    if '<::>' in query_id:
        return query_id.split('<::>')[0]
    return query_id

def main():
    print("="*80)
    print("Run Base MonoT5 on FULL Dataset")
    print("="*80)

    # 1. Load Queries & Results
    print("Loading data...")
    queries_by_strategy = {}
    for strat in QUERY_STRATEGIES:
        queries_by_strategy[strat] = load_queries(strat, project_root)

    retrieval_method = 'elser'
    results_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / retrieval_method / "results"
    
    results_by_strategy = {}
    for strat in QUERY_STRATEGIES:
        # Load ALL results (no filtering)
        raw_results = load_retrieval_results_with_texts(results_dir, retrieval_method, strat)
        results_by_strategy[strat] = raw_results

    # 2. Initialize Scorer
    scorer = MonoT5Scorer(MODEL_NAME, DEVICE)

    # 3. Processing
    # Find tasks common to all strategies
    tasks_to_process = set(results_by_strategy['rewrite'].keys())
    for strat in QUERY_STRATEGIES:
        tasks_to_process &= set(results_by_strategy[strat].keys())
    
    print(f"Processing {len(tasks_to_process)} total tasks (Full Dataset)...")

    output_data = {
        "summary": {},
        "tasks": {}
    }

    correct_selections = 0
    total_gap = 0.0

    for task_id in tqdm(tasks_to_process):
        # Determine Oracle Best
        actual_metrics = {}
        best_r10 = -1.0
        oracle_strat = None
        
        for strat in QUERY_STRATEGIES:
            data = results_by_strategy[strat].get(task_id, {})
            metrics = data.get('retriever_scores', {})
            r10 = metrics.get('recall_10', 0.0)
            actual_metrics[strat] = metrics
            
            if r10 > best_r10:
                best_r10 = r10
                oracle_strat = strat
            elif r10 == best_r10:
                pass

        # Score with MonoT5
        query = queries_by_strategy['rewrite'].get(task_id, "")
        # Fallback query
        if not query:
             for s in QUERY_STRATEGIES:
                 query = queries_by_strategy[s].get(task_id, '')
                 if query: break
        
        if not query: 
            continue

        predicted_recalls = {}
        
        for strat in QUERY_STRATEGIES:
            contexts = results_by_strategy[strat][task_id].get('contexts', [])[:10]
            doc_texts = [c.get('text', '') for c in contexts if c.get('text')]
            
            if not doc_texts:
                predicted_recalls[strat] = 0.0
                continue
                
            scores = scorer.score_batch(query, doc_texts)
            # Calculate predicted Recall
            relevant_count = sum(1 for s in scores if s > 0.5)
            predicted_recalls[strat] = relevant_count / len(scores) if scores else 0.0

        # Select Strategy
        selected_strat = max(predicted_recalls.items(), key=lambda x: x[1])[0]
        
        # Check correctness
        is_optimal = (selected_strat == oracle_strat)
        if is_optimal:
            correct_selections += 1
            
        # Calculate gap
        selected_r10 = actual_metrics[selected_strat].get('recall_10', 0.0)
        gap = best_r10 - selected_r10
        total_gap += gap

        output_data["tasks"][task_id] = {
            "selected_strategy": selected_strat,
            "oracle_strategy": oracle_strat,
            "is_optimal": is_optimal,
            "predicted_recalls": predicted_recalls,
            "actual_metrics": actual_metrics,
            "gap": gap
        }

    # Summary
    n = len(tasks_to_process)
    if n > 0:
        output_data["summary"] = {
            "total_tasks": n,
            "accuracy": correct_selections / n,
            "avg_gap": total_gap / n,
            "model": MODEL_NAME
        }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
