#!/usr/bin/env python3
"""
Run Fine-Tuned MonoT5 for Query Strategy Selection on Test Set.
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
    calculate_predicted_recall_at_k,
    DOMAINS,
    QUERY_STRATEGIES,
)

# Configuration
# Path to the fine-tuned model checkpoint
FINETUNED_MODEL_PATH = project_root / "scripts/ideas/retrieval_tasks/mono-t5-reranker-training/checkpoints/final_model"
TEST_DATA_PATH = project_root / "scripts/ideas/retrieval_tasks/mono-t5-reranker-training/data/test.jsonl"
OUTPUT_FILE = script_dir / "results" / "finetuned_model_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class MonoT5Scorer:
    def __init__(self, model_path: Path, device: str):
        self.device = device
        print(f"Loading fine-tuned model from: {model_path} on {device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
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

def load_test_task_ids(test_data_path: Path) -> Set[str]:
    task_ids = set()
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'task_id' in data:
                    task_ids.add(data['task_id'])
    print(f"Loaded {len(task_ids)} test task IDs")
    return task_ids

def extract_task_id(query_id: str) -> str:
    if '<::>' in query_id:
        return query_id.split('<::>')[0]
    return query_id

def main():
    print("="*80)
    print("Run Fine-Tuned MonoT5 on Test Set")
    print("="*80)

    # 1. Load Test IDs
    try:
        test_task_ids = load_test_task_ids(TEST_DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Load Queries & Results
    print("Loading data...")
    queries_by_strategy = {}
    for strat in QUERY_STRATEGIES:
        queries_by_strategy[strat] = load_queries(strat, project_root)

    retrieval_method = 'elser'
    results_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / retrieval_method / "results"
    
    results_by_strategy = {}
    for strat in QUERY_STRATEGIES:
        raw_results = load_retrieval_results_with_texts(results_dir, retrieval_method, strat)
        # Filter to test set
        results_by_strategy[strat] = {
            qid: data for qid, data in raw_results.items() 
            if extract_task_id(qid) in test_task_ids
        }

    # 3. Initialize Scorer
    scorer = MonoT5Scorer(FINETUNED_MODEL_PATH, DEVICE)

    # 4. Processing
    tasks_to_process = set(results_by_strategy['rewrite'].keys())
    for strat in QUERY_STRATEGIES:
        tasks_to_process &= set(results_by_strategy[strat].keys())
    
    print(f"Processing {len(tasks_to_process)} common test tasks...")

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
        if not query: continue

        predicted_recalls = {}
        
        for strat in QUERY_STRATEGIES:
            contexts = results_by_strategy[strat][task_id].get('contexts', [])[:10]
            doc_texts = [c.get('text', '') for c in contexts if c.get('text')]
            
            if not doc_texts:
                predicted_recalls[strat] = 0.0
                continue
                
            scores = scorer.score_batch(query, doc_texts)
            # Calculate predicted Recall
            # Threshold 0.5 is standard
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
            "model_path": str(FINETUNED_MODEL_PATH)
        }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
