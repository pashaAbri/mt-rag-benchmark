#!/usr/bin/env python3
"""
Filter targeted rewrite results to match the exact query set of the baseline (777 queries)
and re-evaluate to ensure a strict apples-to-apples comparison.
"""

import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import ast

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

BASELINE_RESULTS_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts' / 'elser' / 'results'
TARGETED_RERANK_DIR = script_dir / 'intermediate' / 'using_targeted_rewrite_query'
FILTERED_DIR = script_dir / 'intermediate' / 'filtered_results_777'
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
COMBINATIONS = [
    'lastturn_questions',
    'lastturn_questions_targeted_rewrite',
    'lastturn_targeted_rewrite',
    'questions_targeted_rewrite'
]

# Targeted Rewrite (No Rerank) files to filter too
# These are in a different place: scripts/ideas/retrieval_tasks/targeted_rewrite/retrieval_results/
TARGETED_RAW_DIR = project_root / 'scripts' / 'ideas' / 'retrieval_tasks' / 'targeted_rewrite' / 'retrieval_results'

RETRIEVERS = ['elser', 'bm25', 'bge']

def load_valid_ids(domain):
    """Load valid task IDs from baseline result file."""
    baseline_file = BASELINE_RESULTS_DIR / f"elser_{domain}_rewrite.jsonl"
    valid_ids = set()
    if not baseline_file.exists():
        print(f"Warning: Baseline file not found: {baseline_file}")
        return valid_ids
        
    with open(baseline_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'task_id' in data:
                    valid_ids.add(data['task_id'])
            except:
                pass
    return valid_ids

def filter_file(input_path, output_path, valid_ids):
    """Filter JSONL file to keep only valid_ids."""
    if not input_path.exists():
        print(f"Warning: Input file not found: {input_path}")
        return 0
        
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                data = json.loads(line)
                # handle both '_id' (beir) and 'task_id' (results)
                tid = data.get('task_id') or data.get('_id')
                if tid in valid_ids:
                    fout.write(line)
                    count += 1
            except:
                pass
    return count

def run_eval(input_file, output_file):
    """Run evaluation script."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT_PATH),
        "--input_file", str(input_file),
        "--output_file", str(output_file)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Eval failed for {input_file.name}: {e}")
        return False

def get_metrics(agg_file):
    """Extract metrics from aggregate CSV."""
    if not agg_file.exists():
        return None
    try:
        df = pd.read_csv(agg_file)
        row = df.iloc[0] # Take first row
        
        recall = ast.literal_eval(row['Recall'])
        ndcg = ast.literal_eval(row['nDCG'])
        
        return {
            'R@1': recall[0], 'R@3': recall[1], 'R@5': recall[2], 'R@10': recall[3],
            'nDCG@1': ndcg[0], 'nDCG@3': ndcg[1], 'nDCG@5': ndcg[2], 'nDCG@10': ndcg[3],
            'count': row['count']
        }
    except:
        return None

def main():
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Filtering results to match Baseline 777 Query Set")
    print("="*80)
    
    # Store metrics for summary
    all_metrics = {} # {retriever: {domain: metrics}}
    
    # Filter Targeted Rewrite Raw Results for all retrievers
    for retriever in RETRIEVERS:
        print(f"\nProcessing Targeted Rewrite ({retriever.upper()})...")
        metrics_key = f"Targeted_{retriever}"
        all_metrics[metrics_key] = {}
        
        for domain in DOMAINS:
            valid_ids = load_valid_ids(domain)
            
            raw_input = TARGETED_RAW_DIR / f"targeted_rewrite_{domain}_{retriever}.jsonl"
            if not raw_input.exists():
                print(f"  Skipping {domain}: input file not found")
                continue
                
            raw_output = FILTERED_DIR / f"targeted_rewrite_{domain}_{retriever}_filtered.jsonl"
            raw_eval = FILTERED_DIR / f"targeted_rewrite_{domain}_{retriever}_filtered_evaluated.jsonl"
            
            count = filter_file(raw_input, raw_output, valid_ids)
            # print(f"  Filtered {domain}: kept {count} queries")
            
            if run_eval(raw_output, raw_eval):
                agg_file = raw_eval.with_suffix('')
                agg_file = Path(str(agg_file) + "_aggregate.csv")
                metrics = get_metrics(agg_file)
                if metrics:
                    all_metrics[metrics_key][domain] = metrics

    # Calculate ALL (Weighted Average) for each retriever
    print("\nCalculating ALL metrics...")
    for key in all_metrics:
        total_count = 0
        weighted_metrics = {k: 0.0 for k in all_metrics[key][DOMAINS[0]].keys() if k != 'count'}
        
        for domain in DOMAINS:
            if domain in all_metrics[key]:
                m = all_metrics[key][domain]
                c = m['count']
                total_count += c
                for k in weighted_metrics:
                    weighted_metrics[k] += m[k] * c
        
        if total_count > 0:
            for k in weighted_metrics:
                weighted_metrics[k] /= total_count
            weighted_metrics['count'] = total_count
            all_metrics[key]['ALL'] = weighted_metrics

    # Print Summary Tables per Retriever
    for retriever in RETRIEVERS:
        key = f"Targeted_{retriever}"
        if key not in all_metrics: continue
        
        print("\n" + "="*80)
        print(f"STRICT 777 QUERY SET: {retriever.upper()}")
        print("="*80)
        
        print(f"{'Domain':<10} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'nDCG@1':<8} {'nDCG@3':<8} {'nDCG@5':<8} {'nDCG@10':<8}")
        for domain in ['ALL'] + DOMAINS:
            if domain in all_metrics[key]:
                m = all_metrics[key][domain]
                print(f"{domain:<10} {m['R@1']:.4f}   {m['R@3']:.4f}   {m['R@5']:.4f}   {m['R@10']:.4f}   {m['nDCG@1']:.4f}   {m['nDCG@3']:.4f}   {m['nDCG@5']:.4f}   {m['nDCG@10']:.4f}")

if __name__ == '__main__':
    main()
