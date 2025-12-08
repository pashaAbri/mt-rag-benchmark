import pandas as pd
import json
from pathlib import Path

def extract_turn_from_task_id(task_id):
    """Extract turn number from task_id format: 'base_id<::>turn'"""
    if '<::>' in str(task_id):
        return int(task_id.split('<::>')[1])
    return None

def load_baseline_by_domain(query_type='lastturn', retriever='bge', filter_to_answerable_partial=True):
    """Load baseline results grouped by domain and turn."""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent.parent
    
    # Load baseline task IDs to filter (baselines only include Answerable + Partial)
    baseline_task_ids = set()
    if filter_to_answerable_partial:
        baseline_path = workspace_root / f"scripts/baselines/retrieval_scripts/{retriever}/results/{retriever}_all_{query_type}_evaluated.jsonl"
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
        eval_path = workspace_root / f"scripts/baselines/retrieval_scripts/{retriever}/results/{retriever}_{domain}_{query_type}_evaluated.jsonl"
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
                        'domain': domain,
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
    # Group by domain and turn, then aggregate
    turn_stats = df.groupby(['domain', 'turn']).agg({
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

def load_mmr_by_domain(filter_to_answerable_partial=True):
    """Load MMR clustering results grouped by domain and turn."""
    script_dir = Path(__file__).parent
    mmr_dir = script_dir.parent
    workspace_root = script_dir.parent.parent.parent.parent.parent
    
    # Load baseline task IDs to filter (baselines only include Answerable + Partial)
    baseline_task_ids = set()
    if filter_to_answerable_partial:
        baseline_path = workspace_root / "scripts/baselines/retrieval_scripts/elser/results/elser_all_lastturn_evaluated.jsonl"
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
        eval_path = mmr_dir / f"results_k10/elser_{domain}_mmr_cluster_k10_evaluated.jsonl"
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
                        'domain': domain,
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
    # Group by domain and turn, then aggregate
    turn_stats = df.groupby(['domain', 'turn']).agg({
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
    print("Domain-by-Domain Turn-by-Turn Performance Analysis")
    print("="*80)
    
    domains = ['clapnq', 'cloud', 'fiqa', 'govt']
    query_types = ['lastturn', 'rewrite', 'questions']
    
    # Load all baseline data by domain
    baselines_by_domain = {}
    for query_type in query_types:
        query_name = query_type.replace('lastturn', 'Last Turn').replace('rewrite', 'Query Rewrite').replace('questions', 'Questions')
        baselines_by_domain[query_name] = {}
        
        for domain in domains:
            result = load_baseline_by_domain(query_type=query_type, retriever='elser')
            if result is not None and domain in result.index.get_level_values(0):
                baselines_by_domain[query_name][domain] = result.loc[domain]
    
    # Load MMR data by domain
    mmr_by_domain = {}
    mmr_results = load_mmr_by_domain(filter_to_answerable_partial=True)
    if mmr_results is not None:
        for domain in domains:
            if domain in mmr_results.index.get_level_values(0):
                mmr_by_domain[domain] = mmr_results.loc[domain]
    
    # Create comprehensive data structure
    domain_data = {}
    
    for domain in domains:
        domain_data[domain] = {
            'baselines': {},
            'mmr_cluster': {}
        }
        
        # Add baseline data
        for query_name in baselines_by_domain:
            if domain in baselines_by_domain[query_name]:
                domain_data[domain]['baselines'][query_name] = {}
                for turn in baselines_by_domain[query_name][domain].index:
                    domain_data[domain]['baselines'][query_name][int(turn)] = {
                        'Recall@5': float(baselines_by_domain[query_name][domain].loc[turn, 'Recall@5']),
                        'Recall@10': float(baselines_by_domain[query_name][domain].loc[turn, 'Recall@10']),
                        'NDCG@5': float(baselines_by_domain[query_name][domain].loc[turn, 'NDCG@5']),
                        'NDCG@10': float(baselines_by_domain[query_name][domain].loc[turn, 'NDCG@10']),
                        'count': int(baselines_by_domain[query_name][domain].loc[turn, 'count'])
                    }
        
        # Add MMR data
        if domain in mmr_by_domain:
            domain_data[domain]['mmr_cluster'] = {}
            for turn in mmr_by_domain[domain].index:
                domain_data[domain]['mmr_cluster'][int(turn)] = {
                    'Recall@5': float(mmr_by_domain[domain].loc[turn, 'Recall@5']),
                    'Recall@10': float(mmr_by_domain[domain].loc[turn, 'Recall@10']),
                    'NDCG@5': float(mmr_by_domain[domain].loc[turn, 'NDCG@5']),
                    'NDCG@10': float(mmr_by_domain[domain].loc[turn, 'NDCG@10']),
                    'count': int(mmr_by_domain[domain].loc[turn, 'count'])
                }
    
    # Save to JSON
    script_dir = Path(__file__).parent
    output_file = script_dir / "domain_turn_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(domain_data, f, indent=2)
    
    print(f"\n✓ Domain-by-domain turn-by-turn data saved to: {output_file}")
    
    # Print summary
    print("\n## Summary by Domain\n")
    for domain in domains:
        if domain not in domain_data:
            continue
        
        print(f"\n### {domain.upper()}")
        print("-"*80)
        
        # Get all turns for this domain
        all_turns = set()
        for method_data in domain_data[domain]['baselines'].values():
            all_turns.update(method_data.keys())
        if domain_data[domain]['mmr_cluster']:
            all_turns.update(domain_data[domain]['mmr_cluster'].keys())
        all_turns = sorted(all_turns)
        
        print(f"{'Turn':<6} | {'Last Turn':<12} | {'Query Rewrite':<12} | {'Questions':<12} | {'MMR Cluster':<12}")
        print("-"*80)
        
        for turn in all_turns:
            row = [str(turn)]
            
            # Last Turn
            if turn in domain_data[domain]['baselines'].get('Last Turn', {}):
                row.append(f"{domain_data[domain]['baselines']['Last Turn'][turn]['Recall@10']:.4f}")
            else:
                row.append('N/A')
            
            # Query Rewrite
            if turn in domain_data[domain]['baselines'].get('Query Rewrite', {}):
                row.append(f"{domain_data[domain]['baselines']['Query Rewrite'][turn]['Recall@10']:.4f}")
            else:
                row.append('N/A')
            
            # Questions
            if turn in domain_data[domain]['baselines'].get('Questions', {}):
                row.append(f"{domain_data[domain]['baselines']['Questions'][turn]['Recall@10']:.4f}")
            else:
                row.append('N/A')
            
            # MMR Cluster
            if turn in domain_data[domain]['mmr_cluster']:
                row.append(f"{domain_data[domain]['mmr_cluster'][turn]['Recall@10']:.4f}")
            else:
                row.append('N/A')
            
            print(" | ".join(f"{val:<{len(header)}}" for val, header in zip(row, ['Turn', 'Last Turn', 'Query Rewrite', 'Questions', 'MMR Cluster'])))

if __name__ == "__main__":
    main()

