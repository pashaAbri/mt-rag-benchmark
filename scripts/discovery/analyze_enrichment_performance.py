#!/usr/bin/env python3
"""
Analyze retrieval performance by enrichment subtypes across different query strategies.

This script:
1. Loads enrichment data from cleaned_data/tasks/
2. Loads retrieval results from scripts/baselines/retrieval_scripts/elser/results/
3. Matches tasks with retrieval results by task_id
4. Calculates performance statistics for each enrichment subtype
5. Compares performance across strategies (lastturn, rewrite, questions)
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Metrics to analyze
METRICS = {
    'recall_1': 'R@1',
    'recall_3': 'R@3',
    'recall_5': 'R@5',
    'recall_10': 'R@10',
    'ndcg_cut_1': 'nDCG@1',
    'ndcg_cut_3': 'nDCG@3',
    'ndcg_cut_5': 'nDCG@5',
    'ndcg_cut_10': 'nDCG@10',
}

STRATEGIES = ['lastturn', 'rewrite', 'questions']
DOMAINS = ['clapnq', 'fiqa', 'govt', 'cloud']
RETRIEVAL_METHODS = ['elser', 'bm25', 'bge']


def load_task_enrichments(tasks_dir: Path) -> Dict[str, Dict]:
    """
    Load all task files and extract enrichments.
    
    Returns:
        Dictionary mapping task_id to enrichment data
    """
    enrichments = {}
    
    for domain_dir in tasks_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain = domain_dir.name
        json_files = list(domain_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                task_id = task_data.get('task_id')
                if not task_id:
                    continue
                
                if 'user' in task_data and 'enrichments' in task_data['user']:
                    enrichments[task_id] = {
                        'domain': domain,
                        'enrichments': task_data['user']['enrichments']
                    }
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
    
    return enrichments


def load_retrieval_results(results_dir: Path, retrieval_method: str, strategy: str) -> Dict[str, Dict]:
    """
    Load retrieval results for a specific retrieval method and strategy.
    
    Args:
        results_dir: Path to results directory
        retrieval_method: One of 'elser', 'bm25', 'bge'
        strategy: One of 'lastturn', 'rewrite', 'questions'
    
    Returns:
        Dictionary mapping task_id to retrieval scores
    """
    results = {}
    
    # Load domain-specific files
    for domain in DOMAINS:
        filename = f"{retrieval_method}_{domain}_{strategy}_evaluated.jsonl"
        filepath = results_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    task_id = data.get('task_id')
                    if task_id and 'retriever_scores' in data:
                        results[task_id] = data['retriever_scores']
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return results


def extract_enrichment_subtypes(enrichments: Dict) -> Dict[str, List[str]]:
    """
    Extract all enrichment subtypes from enrichments dict.
    
    Returns:
        Dictionary with keys: 'question_types', 'multi_turn', 'answerability'
    """
    subtypes = {
        'question_types': enrichments.get('Question Type', []),
        'multi_turn': enrichments.get('Multi-Turn', []),
        'answerability': enrichments.get('Answerability', [])
    }
    return subtypes


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values."""
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }
    
    import statistics
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def analyze_enrichment_performance(
    enrichments: Dict[str, Dict],
    retrieval_results: Dict[str, Dict[str, Dict[str, Dict]]]
) -> Dict:
    """
    Analyze performance by enrichment subtype.
    
    Args:
        enrichments: Task enrichments dict
        retrieval_results: Dict mapping retrieval_method -> strategy -> task_id -> scores
    
    Returns:
        Nested dictionary with statistics
    """
    # Structure: enrichment_type -> subtype -> retrieval_method -> strategy -> metric -> stats
    analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    
    # Group tasks by enrichment subtypes
    for task_id, task_data in enrichments.items():
        task_enrichments = task_data['enrichments']
        subtypes = extract_enrichment_subtypes(task_enrichments)
        
        # For each enrichment type
        for enrichment_type, subtype_list in subtypes.items():
            if not subtype_list:
                continue
            
            # For each subtype (a task can have multiple question types)
            for subtype in subtype_list:
                # For each retrieval method
                for retrieval_method in RETRIEVAL_METHODS:
                    if retrieval_method not in retrieval_results:
                        continue
                    
                    # For each strategy
                    for strategy in STRATEGIES:
                        if strategy not in retrieval_results[retrieval_method]:
                            continue
                        if task_id not in retrieval_results[retrieval_method][strategy]:
                            continue
                        
                        scores = retrieval_results[retrieval_method][strategy][task_id]
                        
                        # For each metric
                        for metric_key, metric_name in METRICS.items():
                            if metric_key not in scores:
                                continue
                            
                            value = scores[metric_key]
                            
                            # Store value for later statistics calculation
                            if 'values' not in analysis[enrichment_type][subtype][retrieval_method][strategy][metric_name]:
                                analysis[enrichment_type][subtype][retrieval_method][strategy][metric_name]['values'] = []
                            
                            analysis[enrichment_type][subtype][retrieval_method][strategy][metric_name]['values'].append(value)
    
    # Calculate statistics for each group
    for enrichment_type in analysis:
        for subtype in analysis[enrichment_type]:
            for retrieval_method in analysis[enrichment_type][subtype]:
                for strategy in analysis[enrichment_type][subtype][retrieval_method]:
                    for metric_name in analysis[enrichment_type][subtype][retrieval_method][strategy]:
                        values = analysis[enrichment_type][subtype][retrieval_method][strategy][metric_name]['values']
                        stats = calculate_statistics(values)
                        analysis[enrichment_type][subtype][retrieval_method][strategy][metric_name] = stats
    
    return dict(analysis)


def print_summary_table(analysis: Dict, enrichment_type: str, metric: str = 'R@5'):
    """Print a summary table comparing strategies for each subtype."""
    print(f"\n{'='*100}")
    print(f"{enrichment_type.upper()} - {metric} Performance by Retrieval Method and Strategy")
    print(f"{'='*100}\n")
    
    # Collect all subtypes
    subtypes = sorted(analysis[enrichment_type].keys())
    
    if not subtypes:
        print("No data available.")
        return
    
    # Print header
    print(f"{'Subtype':<25} {'Count':>8} ", end="")
    for retrieval_method in RETRIEVAL_METHODS:
        for strategy in STRATEGIES:
            print(f"{retrieval_method.upper()}-{strategy.upper():>12} ", end="")
    print()
    print("-" * 150)
    
    # Print rows
    for subtype in subtypes:
        # Get count (use any method/strategy, they should be similar)
        count = 0
        for retrieval_method in RETRIEVAL_METHODS:
            if retrieval_method in analysis[enrichment_type][subtype]:
                for strategy in STRATEGIES:
                    if strategy in analysis[enrichment_type][subtype][retrieval_method]:
                        if metric in analysis[enrichment_type][subtype][retrieval_method][strategy]:
                            count = analysis[enrichment_type][subtype][retrieval_method][strategy][metric]['count']
                            break
                if count > 0:
                    break
            if count > 0:
                break
        
        print(f"{subtype:<25} {count:>8} ", end="")
        
        for retrieval_method in RETRIEVAL_METHODS:
            if retrieval_method not in analysis[enrichment_type][subtype]:
                for _ in STRATEGIES:
                    print(f"{'N/A':>20} ", end="")
                continue
            
            for strategy in STRATEGIES:
                if strategy in analysis[enrichment_type][subtype][retrieval_method]:
                    if metric in analysis[enrichment_type][subtype][retrieval_method][strategy]:
                        mean = analysis[enrichment_type][subtype][retrieval_method][strategy][metric]['mean']
                        print(f"{mean:>20.3f} ", end="")
                    else:
                        print(f"{'N/A':>20} ", end="")
                else:
                    print(f"{'N/A':>20} ", end="")
        print()
    
    print()


def print_detailed_table(analysis: Dict, enrichment_type: str, subtype: str):
    """Print detailed metrics for a specific enrichment subtype."""
    print(f"\n{'='*100}")
    print(f"{enrichment_type.upper()} - {subtype} - Detailed Metrics")
    print(f"{'='*100}\n")
    
    # Print header
    print(f"{'Metric':<12} ", end="")
    for strategy in STRATEGIES:
        print(f"{strategy.upper():>15} ", end="")
    print()
    print("-" * 100)
    
    # Print each metric
    for metric_name in METRICS.values():
        print(f"{metric_name:<12} ", end="")
        
        for strategy in STRATEGIES:
            if metric_name in analysis[enrichment_type][subtype][strategy]:
                mean = analysis[enrichment_type][subtype][strategy][metric_name]['mean']
                count = analysis[enrichment_type][subtype][strategy][metric_name]['count']
                print(f"{mean:>8.3f} (n={count:>3}) ", end="")
            else:
                print(f"{'N/A':>15} ", end="")
        print()
    
    print()


def save_results_to_csv(analysis: Dict, output_dir: Path):
    """Save analysis results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for enrichment_type in analysis:
        rows = []
        
        for subtype in analysis[enrichment_type]:
            for retrieval_method in RETRIEVAL_METHODS:
                if retrieval_method not in analysis[enrichment_type][subtype]:
                    continue
                
                for strategy in STRATEGIES:
                    if strategy not in analysis[enrichment_type][subtype][retrieval_method]:
                        continue
                    
                    for metric_name in METRICS.values():
                        if metric_name in analysis[enrichment_type][subtype][retrieval_method][strategy]:
                            stats = analysis[enrichment_type][subtype][retrieval_method][strategy][metric_name]
                            rows.append({
                                'enrichment_type': enrichment_type,
                                'subtype': subtype,
                                'retrieval_method': retrieval_method,
                                'strategy': strategy,
                                'metric': metric_name,
                                'mean': stats['mean'],
                                'median': stats['median'],
                                'std': stats['std'],
                                'min': stats['min'],
                                'max': stats['max'],
                                'count': stats['count']
                            })
        
        if rows:
            filename = f"enrichment_performance_{enrichment_type}.csv"
            filepath = output_dir / filename
            
            # Write CSV using standard library
            with open(filepath, 'w', newline='') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            
            print(f"Saved {filename}")


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    # Go up two levels: scripts/discovery -> scripts -> repo_root
    repo_root = script_dir.parent.parent
    
    tasks_dir = repo_root / "cleaned_data" / "tasks"
    results_dir = repo_root / "scripts" / "baselines" / "retrieval_scripts" / "elser" / "results"
    output_dir = script_dir / "enrichment_analysis_results"
    
    print("="*100)
    print("Enrichment Subtype Performance Analysis")
    print("="*100)
    print(f"\nLoading task enrichments from: {tasks_dir}")
    
    # Load enrichments
    enrichments = load_task_enrichments(tasks_dir)
    print(f"Loaded {len(enrichments)} tasks with enrichments")
    
    # Load retrieval results for each retrieval method and strategy
    print(f"\nLoading retrieval results...")
    retrieval_results = {}
    
    for retrieval_method in RETRIEVAL_METHODS:
        method_results_dir = repo_root / "scripts" / "baselines" / "retrieval_scripts" / retrieval_method / "results"
        if not method_results_dir.exists():
            print(f"Warning: {retrieval_method} results directory not found")
            continue
        
        retrieval_results[retrieval_method] = {}
        print(f"  {retrieval_method.upper()}:")
        
        for strategy in STRATEGIES:
            print(f"    Loading {strategy}...", end=" ")
            results = load_retrieval_results(method_results_dir, retrieval_method, strategy)
            retrieval_results[retrieval_method][strategy] = results
            print(f"{len(results)} tasks")
    
    # Analyze performance
    print("\nAnalyzing performance by enrichment subtype...")
    analysis = analyze_enrichment_performance(enrichments, retrieval_results)
    
    # Print summary tables
    print("\n" + "="*100)
    print("SUMMARY TABLES")
    print("="*100)
    
    # Question Type summary
    print_summary_table(analysis, 'question_types', 'R@5')
    print_summary_table(analysis, 'question_types', 'nDCG@5')
    
    # Multi-Turn summary
    print_summary_table(analysis, 'multi_turn', 'R@5')
    print_summary_table(analysis, 'multi_turn', 'nDCG@5')
    
    # Answerability summary
    print_summary_table(analysis, 'answerability', 'R@5')
    print_summary_table(analysis, 'answerability', 'nDCG@5')
    
    # Save results to CSV
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    save_results_to_csv(analysis, output_dir)
    
    # Print key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    # Find best/worst performing subtypes (using elser-rewrite as reference)
    print("\n1. Question Types - R@5 Performance (ELSER-Rewrite):")
    qtype_r5 = {}
    for subtype in analysis['question_types']:
        if 'elser' in analysis['question_types'][subtype]:
            if 'rewrite' in analysis['question_types'][subtype]['elser']:
                if 'R@5' in analysis['question_types'][subtype]['elser']['rewrite']:
                    qtype_r5[subtype] = analysis['question_types'][subtype]['elser']['rewrite']['R@5']['mean']
    
    if qtype_r5:
        best_qtype = max(qtype_r5.items(), key=lambda x: x[1])
        worst_qtype = min(qtype_r5.items(), key=lambda x: x[1])
        print(f"   Best: {best_qtype[0]} ({best_qtype[1]:.3f})")
        print(f"   Worst: {worst_qtype[0]} ({worst_qtype[1]:.3f})")
    
    print("\n2. Multi-Turn Types - R@5 Performance (ELSER-Rewrite):")
    for subtype in sorted(analysis['multi_turn'].keys()):
        if 'elser' in analysis['multi_turn'][subtype]:
            if 'rewrite' in analysis['multi_turn'][subtype]['elser']:
                if 'R@5' in analysis['multi_turn'][subtype]['elser']['rewrite']:
                    r5 = analysis['multi_turn'][subtype]['elser']['rewrite']['R@5']['mean']
                    count = analysis['multi_turn'][subtype]['elser']['rewrite']['R@5']['count']
                    print(f"   {subtype}: {r5:.3f} (n={count})")
    
    print("\n3. Answerability - R@5 Performance (ELSER-Rewrite):")
    for subtype in sorted(analysis['answerability'].keys()):
        if 'elser' in analysis['answerability'][subtype]:
            if 'rewrite' in analysis['answerability'][subtype]['elser']:
                if 'R@5' in analysis['answerability'][subtype]['elser']['rewrite']:
                    r5 = analysis['answerability'][subtype]['elser']['rewrite']['R@5']['mean']
                    count = analysis['answerability'][subtype]['elser']['rewrite']['R@5']['count']
                    print(f"   {subtype}: {r5:.3f} (n={count})")
    
    # Strategy comparison (using elser)
    print("\n4. Strategy Comparison - ELSER (Average R@5 across all subtypes):")
    strategy_avg = {}
    for strategy in STRATEGIES:
        all_r5 = []
        for enrichment_type in analysis:
            for subtype in analysis[enrichment_type]:
                if 'elser' in analysis[enrichment_type][subtype]:
                    if strategy in analysis[enrichment_type][subtype]['elser']:
                        if 'R@5' in analysis[enrichment_type][subtype]['elser'][strategy]:
                            all_r5.append(analysis[enrichment_type][subtype]['elser'][strategy]['R@5']['mean'])
        if all_r5:
            strategy_avg[strategy] = sum(all_r5) / len(all_r5)
            print(f"   {strategy.upper()}: {strategy_avg[strategy]:.3f}")
    
    # Retrieval method comparison (using rewrite strategy)
    print("\n5. Retrieval Method Comparison - Rewrite Strategy (Average R@5):")
    method_avg = {}
    for retrieval_method in RETRIEVAL_METHODS:
        all_r5 = []
        for enrichment_type in analysis:
            for subtype in analysis[enrichment_type]:
                if retrieval_method in analysis[enrichment_type][subtype]:
                    if 'rewrite' in analysis[enrichment_type][subtype][retrieval_method]:
                        if 'R@5' in analysis[enrichment_type][subtype][retrieval_method]['rewrite']:
                            all_r5.append(analysis[enrichment_type][subtype][retrieval_method]['rewrite']['R@5']['mean'])
        if all_r5:
            method_avg[retrieval_method] = sum(all_r5) / len(all_r5)
            print(f"   {retrieval_method.upper()}: {method_avg[retrieval_method]:.3f}")
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print("\nTo see detailed metrics for a specific subtype, modify the script to call:")
    print("  print_detailed_table(analysis, 'question_types', 'Factoid')")


if __name__ == '__main__':
    main()

