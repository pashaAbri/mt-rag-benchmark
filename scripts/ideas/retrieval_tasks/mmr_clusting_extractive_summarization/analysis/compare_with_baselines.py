import pandas as pd
import json
import os
from pathlib import Path

def parse_array_string(s):
    """Parse string representation of array '[0.1, 0.2, ...]' to list of floats"""
    try:
        clean_s = s.strip('[]')
        if not clean_s:
            return []
        return [float(x.strip()) for x in clean_s.split(',')]
    except Exception as e:
        return []

def load_baseline_results(retriever='bge', query_type='lastturn'):
    """Load baseline results from aggregate CSV files."""
    # Path relative to workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent.parent
    baseline_path = workspace_root / f"scripts/baselines/retrieval_scripts/{retriever}/results/{retriever}_all_{query_type}_evaluated_aggregate.csv"
    
    if not os.path.exists(baseline_path):
        print(f"Warning: Baseline file not found: {baseline_path}")
        return None
    
    df = pd.read_csv(baseline_path)
    # Get the 'all' row
    row = df[df['collection'] == 'all'].iloc[0]
    
    ndcg_scores = parse_array_string(row['nDCG'])
    recall_scores = parse_array_string(row['Recall'])
    
    return {
        'Recall@1': float(recall_scores[0]) if len(recall_scores) > 0 else 0.0,
        'Recall@3': float(recall_scores[1]) if len(recall_scores) > 1 else 0.0,
        'Recall@5': float(recall_scores[2]) if len(recall_scores) > 2 else 0.0,
        'Recall@10': float(recall_scores[3]) if len(recall_scores) > 3 else 0.0,
        'NDCG@1': float(ndcg_scores[0]) if len(ndcg_scores) > 0 else 0.0,
        'NDCG@3': float(ndcg_scores[1]) if len(ndcg_scores) > 1 else 0.0,
        'NDCG@5': float(ndcg_scores[2]) if len(ndcg_scores) > 2 else 0.0,
        'NDCG@10': float(ndcg_scores[3]) if len(ndcg_scores) > 3 else 0.0,
        'count': int(row['count'])
    }

def load_mmr_results_by_turn():
    """Load MMR clustering results grouped by turn."""
    script_dir = Path(__file__).parent
    mmr_dir = script_dir.parent
    
    domains = ['clapnq', 'cloud', 'fiqa', 'govt']
    all_data = []
    
    for domain in domains:
        eval_path = mmr_dir / f"results_k10/bge_{domain}_mmr_cluster_k10_evaluated.jsonl"
        if not os.path.exists(eval_path):
            continue
            
        with open(eval_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    task_id = entry.get('task_id', '')
                    # Extract turn from task_id format: "base_id<::>turn"
                    if '<::>' in task_id:
                        turn = int(task_id.split('<::>')[1])
                    else:
                        continue
                    
                    scores = entry.get('retriever_scores', {})
                    all_data.append({
                        'domain': domain,
                        'turn': turn,
                        'Recall@10': scores.get('recall_10', 0),
                        'NDCG@10': scores.get('ndcg_cut_10', 0),
                    })
                except Exception as e:
                    continue
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    # Group by turn and aggregate
    turn_stats = df.groupby('turn').agg({
        'Recall@10': ['mean', 'count'],
        'NDCG@10': 'mean'
    }).round(4)
    
    # Flatten columns
    turn_stats.columns = ['_'.join(col).strip() for col in turn_stats.columns.values]
    turn_stats.rename(columns={
        'Recall@10_count': 'count',
        'Recall@10_mean': 'Recall@10',
        'NDCG@10_mean': 'NDCG@10'
    }, inplace=True)
    
    return turn_stats

def main():
    print("="*80)
    print("Comparing MMR Clustering (k10, lambda=0.7) with BGE Baselines")
    print("="*80)
    
    # Load baselines
    baselines = {}
    for query_type in ['lastturn', 'rewrite', 'questions']:
        result = load_baseline_results(retriever='bge', query_type=query_type)
        if result:
            baselines[query_type] = result
    
    # Load MMR results by turn
    mmr_by_turn = load_mmr_results_by_turn()
    
    if mmr_by_turn is None:
        print("Error: Could not load MMR results")
        return
    
    # Create comparison table
    print("\n## Overall Comparison (All Turns)")
    print("="*80)
    print(f"{'Method':<25} | {'Recall@10':<12} | {'NDCG@10':<12} | {'Count':<8}")
    print("-"*80)
    
    # Baselines
    for query_type, metrics in baselines.items():
        query_name = query_type.replace('lastturn', 'Last Turn').replace('rewrite', 'Query Rewrite').replace('questions', 'Questions')
        print(f"{query_name:<25} | {metrics['Recall@10']:<12.4f} | {metrics['NDCG@10']:<12.4f} | {metrics['count']:<8}")
    
    # MMR overall (weighted average across all turns)
    mmr_overall_recall = (mmr_by_turn['Recall@10'] * mmr_by_turn['count']).sum() / mmr_by_turn['count'].sum()
    mmr_overall_ndcg = (mmr_by_turn['NDCG@10'] * mmr_by_turn['count']).sum() / mmr_by_turn['count'].sum()
    mmr_total_count = mmr_by_turn['count'].sum()
    print(f"{'MMR Cluster (k10, λ=0.7)':<25} | {mmr_overall_recall:<12.4f} | {mmr_overall_ndcg:<12.4f} | {int(mmr_total_count):<8}")
    
    # Turn-by-turn comparison
    print("\n## Turn-by-Turn Comparison")
    print("="*80)
    
    # Get baseline values for comparison
    baseline_rewrite = baselines.get('rewrite', {})
    
    print(f"\n{'Turn':<6} | {'MMR Recall@10':<15} | {'MMR NDCG@10':<15} | {'vs Rewrite ΔR':<15} | {'vs Rewrite ΔN':<15} | {'Count':<8}")
    print("-"*80)
    
    for turn in sorted(mmr_by_turn.index):
        mmr_recall = mmr_by_turn.loc[turn, 'Recall@10']
        mmr_ndcg = mmr_by_turn.loc[turn, 'NDCG@10']
        count = int(mmr_by_turn.loc[turn, 'count'])
        
        # Compare with rewrite baseline
        delta_r = mmr_recall - baseline_rewrite.get('Recall@10', 0)
        delta_n = mmr_ndcg - baseline_rewrite.get('NDCG@10', 0)
        
        print(f"{int(turn):<6} | {mmr_recall:<15.4f} | {mmr_ndcg:<15.4f} | {delta_r:+.4f}          | {delta_n:+.4f}          | {count:<8}")
    
    # Summary statistics
    print("\n## Summary Statistics")
    print("="*80)
    
    # Best performing turn
    best_turn = mmr_by_turn['Recall@10'].idxmax()
    worst_turn = mmr_by_turn['Recall@10'].idxmin()
    
    print(f"Best Turn: {int(best_turn)} (Recall@10 = {mmr_by_turn.loc[best_turn, 'Recall@10']:.4f})")
    print(f"Worst Turn: {int(worst_turn)} (Recall@10 = {mmr_by_turn.loc[worst_turn, 'Recall@10']:.4f})")
    print(f"\nvs Query Rewrite Baseline:")
    print(f"  Recall@10: {mmr_overall_recall:.4f} vs {baseline_rewrite.get('Recall@10', 0):.4f} (Δ = {mmr_overall_recall - baseline_rewrite.get('Recall@10', 0):+.4f})")
    print(f"  NDCG@10:   {mmr_overall_ndcg:.4f} vs {baseline_rewrite.get('NDCG@10', 0):.4f} (Δ = {mmr_overall_ndcg - baseline_rewrite.get('NDCG@10', 0):+.4f})")
    
    # Save detailed comparison
    # Convert DataFrame to dict with proper type conversion
    mmr_dict = {}
    for turn, row in mmr_by_turn.iterrows():
        mmr_dict[int(turn)] = {
            'Recall@10': float(row['Recall@10']),
            'NDCG@10': float(row['NDCG@10']),
            'count': int(row['count'])
        }
    
    comparison_data = {
        'baselines': baselines,
        'mmr_by_turn': mmr_dict,
        'mmr_overall': {
            'Recall@10': float(mmr_overall_recall),
            'NDCG@10': float(mmr_overall_ndcg),
            'count': int(mmr_total_count)
        }
    }
    
    output_file = "baseline_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Detailed comparison saved to: {output_file}")

if __name__ == "__main__":
    main()

