import argparse
import json
import os
from tqdm import tqdm
from typing import List
from qpp_predictor import QPPPredictor
from utils import load_run_file, load_queries, compute_predicted_ndcg

def run_strategy_selection(domains: List[str], checkpoint_path: str, max_samples: int = None):
    # Adjust path relative to script location
    base_path = "../../../baselines/retrieval_scripts/elser/results/"
    strategies = ["lastturn", "rewrite", "questions"]
    
    predictor = QPPPredictor(checkpoint_path)
    
    os.makedirs("output", exist_ok=True)
    
    for domain in domains:
        print(f"Processing domain: {domain}")
        
        # Load queries
        queries = load_queries(domain)
        if not queries:
            continue
            
        # Load results
        domain_results = {}
        for strat in strategies:
            filename = f"elser_{domain}_{strat}.jsonl"
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                domain_results[strat] = load_run_file(path)
            else:
                print(f"Warning: {path} not found")
        
        if not domain_results:
            continue
            
        task_ids = set(domain_results[strategies[0]].keys())
        for s in strategies[1:]:
            if s in domain_results:
                task_ids &= set(domain_results[s].keys())
        
        # Limit to sample size if specified
        if max_samples:
            task_ids = list(task_ids)[:max_samples]
            print(f"Limiting to {max_samples} queries for testing")
        
        out_file = f"output/selection_results_{domain}.jsonl"
        
        with open(out_file, 'w') as f_out:
            for tid in tqdm(task_ids, desc=f"Evaluating {domain}"):
                query_text = queries.get(tid)
                if not query_text:
                    continue
                    
                strategy_scores = {}
                
                for strat in strategies:
                    if strat not in domain_results: continue
                    
                    contexts = domain_results[strat][tid]
                    passages = [c.get('text', '') for c in contexts]
                    
                    # Predict relevance
                    preds = predictor.predict(query_text, passages)
                    
                    # Compute nDCG@10
                    score = compute_predicted_ndcg(preds)
                    strategy_scores[strat] = score
                
                # Select best strategy
                best_strat = max(strategy_scores, key=strategy_scores.get) if strategy_scores else "none"
                
                result = {
                    "task_id": tid,
                    "selected_strategy": best_strat,
                    "scores": strategy_scores
                }
                f_out.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint/msmarco-v1-passage-dev-small.original-bm25-1000.original-Meta-Llama-3-8B-Instruct-neg2-top1000/checkpoint-2675")
    parser.add_argument("--domains", nargs="+", default=["clapnq"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of queries to process (for testing)")
    args = parser.parse_args()
    
    run_strategy_selection(args.domains, args.checkpoint, args.max_samples)
