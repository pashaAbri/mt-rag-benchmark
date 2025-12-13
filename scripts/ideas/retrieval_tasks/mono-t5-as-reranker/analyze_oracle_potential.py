#!/usr/bin/env python3
"""
Oracle Analysis: Calculate the upper bound performance if we could perfectly select
the best strategy for each query.

Strategies included:
1. Baselines (Retrieval only): lastturn, rewrite
2. Reranked Combinations: lastturn_questions, lastturn_rewrite, questions_rewrite, all_3
"""

import sys
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
INTERMEDIATE_DIR = script_dir / "intermediate"
RERANKED_DIR = INTERMEDIATE_DIR / "using_rewrite_query"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Map friendly names to file patterns
# (Location, Filename Pattern)
STRATEGIES = {
    'Baseline: Lastturn': (INTERMEDIATE_DIR, 'baseline_lastturn_{domain}_evaluated.jsonl'),
    'Baseline: Rewrite': (INTERMEDIATE_DIR, 'baseline_rewrite_{domain}_evaluated.jsonl'),
    'Rerank: L+Q': (RERANKED_DIR, 'reranked_lastturn_questions_{domain}_evaluated.jsonl'),
    'Rerank: L+R': (RERANKED_DIR, 'reranked_lastturn_rewrite_{domain}_evaluated.jsonl'),
    'Rerank: Q+R': (RERANKED_DIR, 'reranked_questions_rewrite_{domain}_evaluated.jsonl'),
    'Rerank: All 3': (RERANKED_DIR, 'reranked_lastturn_questions_rewrite_{domain}_evaluated.jsonl'),
}

def load_metrics(filepath):
    """Load nDCG@10 for each task_id from a jsonl file."""
    metrics = {}
    if not filepath.exists():
        return metrics
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                # 'metrics' usually contains {'nDCG@10': ...} or similar structure depending on eval script output
                # The standard evaluation script output usually looks like:
                # {"task_id": "...", "metrics": {"nDCG@10": 0.5, ...}, ...}
                # OR it might just be the list of metrics.
                # Let's check the structure carefully.
                
                # Based on previous output, the file contains "metrics" key?
                # Actually, standard run_retrieval_eval.py output format:
                # It merges input with metrics.
                # Key typically: "nDCG" -> [val_at_1, val_at_3, val_at_5, val_at_10]
                
                if 'retriever_scores' in data and 'ndcg_cut_10' in data['retriever_scores']:
                    metrics[task_id] = float(data['retriever_scores']['ndcg_cut_10'])
                elif 'nDCG' in data and isinstance(data['nDCG'], list):
                    # It's a list: [at_1, at_3, at_5, at_10]
                    ndcg_10 = data['nDCG'][3] if len(data['nDCG']) > 3 else 0.0
                    metrics[task_id] = float(ndcg_10)
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
    return metrics

def main():
    print("="*80)
    print("Oracle Analysis: Upper Bound Performance Potential")
    print("="*80)
    
    summary_data = []
    
    for domain in DOMAINS:
        print(f"\nAnalyzing domain: {domain.upper()}")
        
        # Store all scores: {task_id: {strategy: score}}
        task_scores = defaultdict(dict)
        
        # Load data for all strategies
        available_strategies = []
        for strat_name, (dir_path, pattern) in STRATEGIES.items():
            filename = pattern.format(domain=domain)
            filepath = dir_path / filename
            
            if not filepath.exists():
                print(f"  Warning: Missing {filename}")
                continue
                
            scores = load_metrics(filepath)
            print(f"  Loaded {len(scores)} queries from {strat_name}")
            
            if scores:
                available_strategies.append(strat_name)
                for tid, score in scores.items():
                    task_scores[tid][strat_name] = score
        
        # Compute Oracle stats
        oracle_scores = []
        best_static_scores = []
        best_static_strategy = None
        
        # Find best static strategy (highest average)
        avg_scores = {}
        for strat in available_strategies:
            strat_values = [scores[strat] for tid, scores in task_scores.items() if strat in scores]
            if strat_values:
                avg_scores[strat] = np.mean(strat_values)
        
        if not avg_scores:
            print("  No data available.")
            continue
            
        best_static_strategy = max(avg_scores, key=avg_scores.get)
        best_static_avg = avg_scores[best_static_strategy]
        
        # Calculate Oracle
        for tid, strategies in task_scores.items():
            if not strategies:
                continue
            oracle_scores.append(max(strategies.values()))
            
        oracle_avg = np.mean(oracle_scores)
        improvement = oracle_avg - best_static_avg
        pct_improvement = (improvement / best_static_avg) * 100 if best_static_avg > 0 else 0.0
        
        print(f"  Best Static: {best_static_strategy} ({best_static_avg:.4f})")
        print(f"  Oracle:      {oracle_avg:.4f}")
        print(f"  Potential:   +{pct_improvement:.2f}%")
        
        summary_data.append({
            'Domain': domain,
            'Best Static Strategy': best_static_strategy,
            'Best Static nDCG@10': f"{best_static_avg:.4f}",
            'Oracle nDCG@10': f"{oracle_avg:.4f}",
            'Gain (Absolute)': f"{improvement:.4f}",
            'Gain (%)': f"{pct_improvement:.2f}%"
        })

    # Print Summary Table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + "="*100)
        print("ORACLE ANALYSIS SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        
        # Suggestion
        print("\nInterpretation:")
        print("- 'Oracle nDCG@10' is the theoretical maximum if we perfectly picked the best strategy per query.")
        print("- 'Gain' shows how much performance is left on the table by using a single static strategy.")
        print("- High gain suggests a query classifier/router could be very valuable.")
        print("- Low gain suggests the best static strategy is already near-optimal or strategies are highly correlated.")

if __name__ == '__main__':
    main()
