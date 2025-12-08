import pandas as pd
import json
from pathlib import Path

def extract_turn_from_task_id(task_id):
    """Extract turn number from task_id format: 'base_id<::>turn'"""
    if '<::>' in str(task_id):
        return int(task_id.split('<::>')[1])
    return None

def load_baseline_by_turn(query_type='lastturn', retriever='bge'):
    """Load baseline results grouped by turn."""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent.parent
    eval_path = workspace_root / f"scripts/baselines/retrieval_scripts/{retriever}/results/{retriever}_all_{query_type}_evaluated.jsonl"
    
    if not eval_path.exists():
        print(f"Warning: File not found: {eval_path}")
        return None
    
    data = []
    with open(eval_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                task_id = entry.get('task_id', '')
                turn = extract_turn_from_task_id(task_id)
                
                if turn is None:
                    continue
                
                scores = entry.get('retriever_scores', {})
                data.append({
                    'turn': turn,
                    'Recall@5': scores.get('recall_5', 0),
                    'Recall@10': scores.get('recall_10', 0),
                    'NDCG@5': scores.get('ndcg_cut_5', 0),
                    'NDCG@10': scores.get('ndcg_cut_10', 0),
                })
            except Exception as e:
                continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    # Group by turn and aggregate
    turn_stats = df.groupby('turn').agg({
        'Recall@5': 'mean',
        'Recall@10': ['mean', 'count'],
        'NDCG@5': 'mean',
        'NDCG@10': 'mean'
    }).round(4)
    
    # Flatten columns
    turn_stats.columns = ['_'.join(col).strip() for col in turn_stats.columns.values]
    turn_stats.rename(columns={
        'Recall@10_count': 'count',
        'Recall@5_mean': 'Recall@5',
        'Recall@10_mean': 'Recall@10',
        'NDCG@5_mean': 'NDCG@5',
        'NDCG@10_mean': 'NDCG@10'
    }, inplace=True)
    
    return turn_stats

def load_mmr_by_turn(filter_to_answerable_partial=True):
    """Load MMR clustering results grouped by turn.
    
    Args:
        filter_to_answerable_partial: If True, only include tasks that are in baseline files
                                      (which contain only Answerable + Partial tasks)
    """
    script_dir = Path(__file__).parent
    mmr_dir = script_dir.parent
    workspace_root = script_dir.parent.parent.parent.parent.parent
    
    # Load baseline task IDs to filter (baselines only include Answerable + Partial)
    baseline_task_ids = set()
    if filter_to_answerable_partial:
        baseline_path = workspace_root / "scripts/baselines/retrieval_scripts/bge/results/bge_all_lastturn_evaluated.jsonl"
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        task_id = entry.get('task_id', '')
                        if task_id:
                            baseline_task_ids.add(task_id)
                    except:
                        continue
    
    domains = ['clapnq', 'cloud', 'fiqa', 'govt']
    all_data = []
    
    for domain in domains:
        eval_path = mmr_dir / f"results_k10/bge_{domain}_mmr_cluster_k10_evaluated.jsonl"
        if not eval_path.exists():
            continue
            
        with open(eval_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    task_id = entry.get('task_id', '')
                    
                    # Filter to only Answerable/Partial tasks if requested
                    if filter_to_answerable_partial and baseline_task_ids:
                        if task_id not in baseline_task_ids:
                            continue
                    
                    turn = extract_turn_from_task_id(task_id)
                    
                    if turn is None:
                        continue
                    
                    scores = entry.get('retriever_scores', {})
                    all_data.append({
                        'turn': turn,
                        'Recall@5': scores.get('recall_5', 0),
                        'Recall@10': scores.get('recall_10', 0),
                        'NDCG@5': scores.get('ndcg_cut_5', 0),
                        'NDCG@10': scores.get('ndcg_cut_10', 0),
                    })
                except Exception as e:
                    continue
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    # Group by turn and aggregate
    turn_stats = df.groupby('turn').agg({
        'Recall@5': 'mean',
        'Recall@10': ['mean', 'count'],
        'NDCG@5': 'mean',
        'NDCG@10': 'mean'
    }).round(4)
    
    # Flatten columns
    turn_stats.columns = ['_'.join(col).strip() for col in turn_stats.columns.values]
    turn_stats.rename(columns={
        'Recall@10_count': 'count',
        'Recall@5_mean': 'Recall@5',
        'Recall@10_mean': 'Recall@10',
        'NDCG@5_mean': 'NDCG@5',
        'NDCG@10_mean': 'NDCG@10'
    }, inplace=True)
    
    return turn_stats

def main():
    print("="*80)
    print("Turn-by-Turn Analysis: Baselines vs MMR Clustering")
    print("="*80)
    
    # Load all baselines
    baselines = {}
    for query_type in ['lastturn', 'rewrite', 'questions']:
        result = load_baseline_by_turn(query_type=query_type)
        if result is not None:
            query_name = query_type.replace('lastturn', 'Last Turn').replace('rewrite', 'Query Rewrite').replace('questions', 'Questions')
            baselines[query_name] = result
    
    # Load MMR results (filtered to Answerable + Partial only for fair comparison)
    mmr_results = load_mmr_by_turn(filter_to_answerable_partial=True)
    
    if mmr_results is None:
        print("Error: Could not load MMR results")
        return
    
    # Print individual baseline analyses
    print("\n## Individual Baseline Turn-by-Turn Analysis\n")
    
    for query_name, stats in baselines.items():
        print(f"### {query_name}")
        print("-"*80)
        print(f"{'Turn':<6} | {'Recall@5':<12} | {'Recall@10':<12} | {'NDCG@5':<12} | {'NDCG@10':<12} | {'Count':<8}")
        print("-"*100)
        
        for turn in sorted(stats.index):
            recall5 = stats.loc[turn, 'Recall@5']
            recall10 = stats.loc[turn, 'Recall@10']
            ndcg5 = stats.loc[turn, 'NDCG@5']
            ndcg10 = stats.loc[turn, 'NDCG@10']
            count = int(stats.loc[turn, 'count'])
            print(f"{int(turn):<6} | {recall5:<12.4f} | {recall10:<12.4f} | {ndcg5:<12.4f} | {ndcg10:<12.4f} | {count:<8}")
        
        # Summary stats
        best_turn = stats['Recall@10'].idxmax()
        worst_turn = stats['Recall@10'].idxmin()
        overall_recall5 = (stats['Recall@5'] * stats['count']).sum() / stats['count'].sum()
        overall_recall10 = (stats['Recall@10'] * stats['count']).sum() / stats['count'].sum()
        overall_ndcg5 = (stats['NDCG@5'] * stats['count']).sum() / stats['count'].sum()
        overall_ndcg10 = (stats['NDCG@10'] * stats['count']).sum() / stats['count'].sum()
        
        print(f"\nSummary:")
        print(f"  Overall: Recall@5 = {overall_recall5:.4f}, Recall@10 = {overall_recall10:.4f}, NDCG@5 = {overall_ndcg5:.4f}, NDCG@10 = {overall_ndcg10:.4f}")
        print(f"  Best Turn: {int(best_turn)} (Recall@10 = {stats.loc[best_turn, 'Recall@10']:.4f})")
        print(f"  Worst Turn: {int(worst_turn)} (Recall@10 = {stats.loc[worst_turn, 'Recall@10']:.4f})")
        print()
    
    # Print MMR analysis
    print("### MMR Cluster (k10, λ=0.7)")
    print("-"*100)
    print(f"{'Turn':<6} | {'Recall@5':<12} | {'Recall@10':<12} | {'NDCG@5':<12} | {'NDCG@10':<12} | {'Count':<8}")
    print("-"*100)
    
    for turn in sorted(mmr_results.index):
        recall5 = mmr_results.loc[turn, 'Recall@5']
        recall10 = mmr_results.loc[turn, 'Recall@10']
        ndcg5 = mmr_results.loc[turn, 'NDCG@5']
        ndcg10 = mmr_results.loc[turn, 'NDCG@10']
        count = int(mmr_results.loc[turn, 'count'])
        print(f"{int(turn):<6} | {recall5:<12.4f} | {recall10:<12.4f} | {ndcg5:<12.4f} | {ndcg10:<12.4f} | {count:<8}")
    
    mmr_best_turn = mmr_results['Recall@10'].idxmax()
    mmr_worst_turn = mmr_results['Recall@10'].idxmin()
    mmr_overall_recall5 = (mmr_results['Recall@5'] * mmr_results['count']).sum() / mmr_results['count'].sum()
    mmr_overall_recall10 = (mmr_results['Recall@10'] * mmr_results['count']).sum() / mmr_results['count'].sum()
    mmr_overall_ndcg5 = (mmr_results['NDCG@5'] * mmr_results['count']).sum() / mmr_results['count'].sum()
    mmr_overall_ndcg10 = (mmr_results['NDCG@10'] * mmr_results['count']).sum() / mmr_results['count'].sum()
    
    print(f"\nSummary:")
    print(f"  Overall: Recall@5 = {mmr_overall_recall5:.4f}, Recall@10 = {mmr_overall_recall10:.4f}, NDCG@5 = {mmr_overall_ndcg5:.4f}, NDCG@10 = {mmr_overall_ndcg10:.4f}")
    print(f"  Best Turn: {int(mmr_best_turn)} (Recall@10 = {mmr_results.loc[mmr_best_turn, 'Recall@10']:.4f})")
    print(f"  Worst Turn: {int(mmr_worst_turn)} (Recall@10 = {mmr_results.loc[mmr_worst_turn, 'Recall@10']:.4f})")
    print()
    
    # Combined comparison table
    print("\n## Combined Comparison: All Methods Turn-by-Turn")
    print("="*80)
    
    # Get all unique turns
    all_turns = set()
    for stats in baselines.values():
        all_turns.update(stats.index)
    all_turns.update(mmr_results.index)
    all_turns = sorted(all_turns)
    
    # Create comparison table
    print(f"\n{'Turn':<6} | {'Last Turn':<15} | {'Query Rewrite':<15} | {'Questions':<15} | {'MMR Cluster':<15}")
    print(f"{'':6} | {'R@10':<7} {'N@10':<7} | {'R@10':<7} {'N@10':<7} | {'R@10':<7} {'N@10':<7} | {'R@10':<7} {'N@10':<7}")
    print("-"*80)
    
    for turn in all_turns:
        row = [f"{int(turn):<6}"]
        
        # Last Turn
        if turn in baselines['Last Turn'].index:
            lt_r = baselines['Last Turn'].loc[turn, 'Recall@10']
            lt_n = baselines['Last Turn'].loc[turn, 'NDCG@10']
            row.append(f"{lt_r:<7.4f} {lt_n:<7.4f}")
        else:
            row.append(f"{'N/A':<7} {'N/A':<7}")
        
        # Query Rewrite
        if turn in baselines['Query Rewrite'].index:
            qr_r = baselines['Query Rewrite'].loc[turn, 'Recall@10']
            qr_n = baselines['Query Rewrite'].loc[turn, 'NDCG@10']
            row.append(f"{qr_r:<7.4f} {qr_n:<7.4f}")
        else:
            row.append(f"{'N/A':<7} {'N/A':<7}")
        
        # Questions
        if turn in baselines['Questions'].index:
            q_r = baselines['Questions'].loc[turn, 'Recall@10']
            q_n = baselines['Questions'].loc[turn, 'NDCG@10']
            row.append(f"{q_r:<7.4f} {q_n:<7.4f}")
        else:
            row.append(f"{'N/A':<7} {'N/A':<7}")
        
        # MMR Cluster
        if turn in mmr_results.index:
            mmr_r = mmr_results.loc[turn, 'Recall@10']
            mmr_n = mmr_results.loc[turn, 'NDCG@10']
            row.append(f"{mmr_r:<7.4f} {mmr_n:<7.4f}")
        else:
            row.append(f"{'N/A':<7} {'N/A':<7}")
        
        print(" | ".join(row))
    
    # Save detailed data
    comparison_data = {
        'baselines': {},
        'mmr_cluster': {}
    }
    
    for query_name, stats in baselines.items():
        comparison_data['baselines'][query_name] = {
            int(turn): {
                'Recall@5': float(stats.loc[turn, 'Recall@5']),
                'Recall@10': float(stats.loc[turn, 'Recall@10']),
                'NDCG@5': float(stats.loc[turn, 'NDCG@5']),
                'NDCG@10': float(stats.loc[turn, 'NDCG@10']),
                'count': int(stats.loc[turn, 'count'])
            }
            for turn in stats.index
        }
    
    comparison_data['mmr_cluster'] = {
        int(turn): {
            'Recall@5': float(mmr_results.loc[turn, 'Recall@5']),
            'Recall@10': float(mmr_results.loc[turn, 'Recall@10']),
            'NDCG@5': float(mmr_results.loc[turn, 'NDCG@5']),
            'NDCG@10': float(mmr_results.loc[turn, 'NDCG@10']),
            'count': int(mmr_results.loc[turn, 'count'])
        }
        for turn in mmr_results.index
    }
    
    script_dir = Path(__file__).parent
    output_file = script_dir / "baselines_turn_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Detailed turn-by-turn data saved to: {output_file}")

if __name__ == "__main__":
    main()

