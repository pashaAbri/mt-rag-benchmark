import pandas as pd
import json
import glob
import os
import re

def parse_array_string(s):
    """Parse string representation of array '[0.1, 0.2, ...]' to list of floats"""
    try:
        # Remove brackets and split by comma
        clean_s = s.strip('[]')
        if not clean_s:
            return []
        return [float(x.strip()) for x in clean_s.split(',')]
    except Exception as e:
        print(f"Error parsing: {s}")
        return []

def load_results(pattern):
    results = {}
    files = glob.glob(pattern)
    
    for file_path in files:
        # Extract domain from filename
        # formats: bm25_DOMAIN_rewrite_evaluated_aggregate.csv or bm25_DOMAIN_mmr_cluster_evaluated_aggregate.csv
        match = re.search(r'bm25_([a-z]+)_(rewrite|mmr_cluster)_evaluated_aggregate\.csv', os.path.basename(file_path))
        if match:
            domain = match.group(1)
            df = pd.read_csv(file_path)
            
            # Get the 'all' row
            row = df[df['collection'] == 'all'].iloc[0]
            
            ndcg_scores = parse_array_string(row['nDCG'])
            recall_scores = parse_array_string(row['Recall'])
            
            # We'll use Recall@10 (index 3) and NDCG@10 (index 3) for comparison
            # Assuming array is [k=1, k=3, k=5, k=10] based on evaluation script
            
            results[domain] = {
                'Recall@10': recall_scores[3] if len(recall_scores) > 3 else 0,
                'NDCG@10': ndcg_scores[3] if len(ndcg_scores) > 3 else 0,
                'Count': row['count']
            }
            
    return results

def main():
    # Paths
    baseline_pattern = "scripts/baselines/retrieval_scripts/bm25/results/bm25_*_rewrite_evaluated_aggregate.csv"
    mmr_pattern = "scripts/ideas/retrieval_tasks/mmr_clusting_extractive_summarization/results/bm25_*_mmr_cluster_evaluated_aggregate.csv"
    
    print("Loading Baseline Results (Standard Rewrite)...")
    baseline_results = load_results(baseline_pattern)
    
    print("Loading MMR Results (Clustering Rewrite)...")
    mmr_results = load_results(mmr_pattern)
    
    # Print Comparison Table
    print("\n" + "="*85)
    print(f"{'Domain':<10} | {'Metric':<10} | {'Baseline':<10} | {'MMR Cluster':<12} | {'Diff':<10} | {'% Change':<10}")
    print("-" * 85)
    
    domains = sorted(list(set(list(baseline_results.keys()) + list(mmr_results.keys()))))
    
    averages = {'Baseline_R': 0, 'MMR_R': 0, 'Baseline_N': 0, 'MMR_N': 0, 'Count': 0}
    
    for domain in domains:
        base = baseline_results.get(domain, {})
        mmr = mmr_results.get(domain, {})
        
        base_r = base.get('Recall@10', 0)
        mmr_r = mmr.get('Recall@10', 0)
        
        base_n = base.get('NDCG@10', 0)
        mmr_n = mmr.get('NDCG@10', 0)
        
        # Calculate diffs
        diff_r = mmr_r - base_r
        pct_r = (diff_r / base_r * 100) if base_r > 0 else 0
        
        diff_n = mmr_n - base_n
        pct_n = (diff_n / base_n * 100) if base_n > 0 else 0
        
        # Update averages
        averages['Baseline_R'] += base_r
        averages['MMR_R'] += mmr_r
        averages['Baseline_N'] += base_n
        averages['MMR_N'] += mmr_n
        averages['Count'] += 1
        
        print(f"{domain:<10} | {'Recall@10':<10} | {base_r:.4f}     | {mmr_r:.4f}       | {diff_r:+.4f}     | {pct_r:+.2f}%")
        print(f"{'':<10} | {'NDCG@10':<10} | {base_n:.4f}     | {mmr_n:.4f}       | {diff_n:+.4f}     | {pct_n:+.2f}%")
        print("-" * 85)
        
    # Print Averages
    if averages['Count'] > 0:
        avg_base_r = averages['Baseline_R'] / averages['Count']
        avg_mmr_r = averages['MMR_R'] / averages['Count']
        avg_base_n = averages['Baseline_N'] / averages['Count']
        avg_mmr_n = averages['MMR_N'] / averages['Count']
        
        diff_avg_r = avg_mmr_r - avg_base_r
        pct_avg_r = (diff_avg_r / avg_base_r * 100) if avg_base_r > 0 else 0
        
        diff_avg_n = avg_mmr_n - avg_base_n
        pct_avg_n = (diff_avg_n / avg_base_n * 100) if avg_base_n > 0 else 0
        
        print(f"{'AVERAGE':<10} | {'Recall@10':<10} | {avg_base_r:.4f}     | {avg_mmr_r:.4f}       | {diff_avg_r:+.4f}     | {pct_avg_r:+.2f}%")
        print(f"{'':<10} | {'NDCG@10':<10} | {avg_base_n:.4f}     | {avg_mmr_n:.4f}       | {diff_avg_n:+.4f}     | {pct_avg_n:+.2f}%")
        print("=" * 85)

if __name__ == "__main__":
    main()

