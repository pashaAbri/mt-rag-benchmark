#!/usr/bin/env python3
"""
Oracle Retriever Analysis: What if we could always pick the best retriever?

This script analyzes what would happen if we had an oracle that tells us
which retriever (bm25, bge, or elser) to use for each query/turn.

For each task_id, we determine which retriever performed best, then calculate
what the aggregate performance would be if we always used the oracle's choice.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import statistics

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
RETRIEVAL_METHODS = ['bm25', 'bge', 'elser']

# Primary metric for oracle selection (can be changed)
PRIMARY_METRIC = 'ndcg_cut_5'  # nDCG@5 is used as the oracle selection metric


def load_retrieval_results(results_dir: Path, retrieval_method: str, strategy: str) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Load retrieval results for a specific retrieval method and strategy.
    
    Args:
        results_dir: Path to results directory (e.g., scripts/baselines/retrieval_scripts/{method}/results)
        retrieval_method: One of 'bm25', 'bge', 'elser'
        strategy: One of 'lastturn', 'rewrite', 'questions'
    
    Returns:
        Tuple of:
        - Dictionary mapping task_id to retrieval scores
        - Dictionary mapping task_id to domain
    """
    results = {}
    task_domains = {}
    
    # Load domain-specific files
    for domain in DOMAINS:
        filename = f"{retrieval_method}_{domain}_{strategy}_evaluated.jsonl"
        filepath = results_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found at {filepath}")
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
                        task_domains[task_id] = domain
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return results, task_domains


def find_oracle_choice(
    task_id: str,
    bm25_scores: Dict[str, float],
    bge_scores: Dict[str, float],
    elser_scores: Dict[str, float],
    metric: str
) -> Tuple[str, float]:
    """
    Determine which retriever performs best for a given task and metric.
    
    Returns:
        Tuple of (best_retriever_name, best_score)
    """
    scores = {}
    if bm25_scores and metric in bm25_scores:
        scores['bm25'] = bm25_scores[metric]
    if bge_scores and metric in bge_scores:
        scores['bge'] = bge_scores[metric]
    if elser_scores and metric in elser_scores:
        scores['elser'] = elser_scores[metric]
    
    if not scores:
        return None, 0.0
    
    best_retriever = max(scores.items(), key=lambda x: x[1])
    return best_retriever[0], best_retriever[1]


def calculate_oracle_performance(
    bm25_results: Dict[str, Dict],
    bge_results: Dict[str, Dict],
    elser_results: Dict[str, Dict],
    metric: str
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[float]]]:
    """
    Calculate oracle performance by always picking the best retriever for each task.
    
    Returns:
        Tuple of:
        - oracle_scores: Dictionary mapping metric to oracle score
        - oracle_choices: Dictionary mapping retriever to count of times chosen
        - per_task_scores: Dictionary mapping retriever to list of scores for tasks where it was chosen
    """
    # Get all task_ids that appear in at least one retriever
    all_task_ids = set()
    all_task_ids.update(bm25_results.keys())
    all_task_ids.update(bge_results.keys())
    all_task_ids.update(elser_results.keys())
    
    oracle_choices = defaultdict(int)
    oracle_scores_by_metric = defaultdict(list)
    per_task_scores = defaultdict(lambda: defaultdict(list))
    
    for task_id in all_task_ids:
        bm25_scores = bm25_results.get(task_id, {})
        bge_scores = bge_results.get(task_id, {})
        elser_scores = elser_results.get(task_id, {})
        
        # Find oracle choice for primary metric
        best_retriever, best_score = find_oracle_choice(
            task_id, bm25_scores, bge_scores, elser_scores, metric
        )
        
        if best_retriever is None:
            continue
        
        oracle_choices[best_retriever] += 1
        
        # Get scores for all metrics using the oracle's choice
        chosen_scores = None
        if best_retriever == 'bm25':
            chosen_scores = bm25_scores
        elif best_retriever == 'bge':
            chosen_scores = bge_scores
        elif best_retriever == 'elser':
            chosen_scores = elser_scores
        
        if chosen_scores:
            for m in METRICS.keys():
                if m in chosen_scores:
                    oracle_scores_by_metric[m].append(chosen_scores[m])
                    per_task_scores[best_retriever][m].append(chosen_scores[m])
    
    # Calculate averages
    oracle_avg_scores = {}
    for m, scores in oracle_scores_by_metric.items():
        if scores:
            oracle_avg_scores[m] = statistics.mean(scores)
        else:
            oracle_avg_scores[m] = 0.0
    
    return oracle_avg_scores, dict(oracle_choices), dict(per_task_scores)


def calculate_individual_performance(results: Dict[str, Dict]) -> Dict[str, float]:
    """Calculate average performance for a single retriever."""
    metric_scores = defaultdict(list)
    
    for task_id, scores in results.items():
        for metric in METRICS.keys():
            if metric in scores:
                metric_scores[metric].append(scores[metric])
    
    avg_scores = {}
    for metric, scores in metric_scores.items():
        if scores:
            avg_scores[metric] = statistics.mean(scores)
        else:
            avg_scores[metric] = 0.0
    
    return avg_scores


def analyze_by_domain(
    bm25_results: Dict[str, Dict],
    bge_results: Dict[str, Dict],
    elser_results: Dict[str, Dict],
    bm25_domains: Dict[str, str],
    bge_domains: Dict[str, str],
    elser_domains: Dict[str, str],
    metric: str
) -> Dict[str, Dict]:
    """Analyze oracle performance by domain."""
    domain_results = defaultdict(lambda: {
        'bm25': {},
        'bge': {},
        'elser': {},
        'oracle': {}
    })
    
    # Group tasks by domain
    domain_tasks = defaultdict(list)
    all_task_ids = set(list(bm25_results.keys()) + list(bge_results.keys()) + list(elser_results.keys()))
    
    for task_id in all_task_ids:
        # Try to get domain from any of the domain mappings
        domain = (bm25_domains.get(task_id) or 
                 bge_domains.get(task_id) or 
                 elser_domains.get(task_id))
        if domain:
            domain_tasks[domain].append(task_id)
    
    for domain, task_ids in domain_tasks.items():
        # Filter results for this domain
        bm25_domain = {tid: bm25_results[tid] for tid in task_ids if tid in bm25_results}
        bge_domain = {tid: bge_results[tid] for tid in task_ids if tid in bge_results}
        elser_domain = {tid: elser_results[tid] for tid in task_ids if tid in elser_results}
        
        # Calculate individual performances
        domain_results[domain]['bm25'] = calculate_individual_performance(bm25_domain)
        domain_results[domain]['bge'] = calculate_individual_performance(bge_domain)
        domain_results[domain]['elser'] = calculate_individual_performance(elser_domain)
        
        # Calculate oracle performance
        oracle_scores, oracle_choices, _ = calculate_oracle_performance(
            bm25_domain, bge_domain, elser_domain, metric
        )
        domain_results[domain]['oracle'] = oracle_scores
        domain_results[domain]['oracle_choices'] = oracle_choices
    
    return dict(domain_results)


def generate_markdown_report(
    strategy: str,
    bm25_perf: Dict[str, float],
    bge_perf: Dict[str, float],
    elser_perf: Dict[str, float],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int],
    domain_analysis: Dict[str, Dict],
    output_file: Path
):
    """Generate a markdown report with the analysis results."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Oracle Retriever Analysis: {strategy.upper()} Strategy\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis explores what would happen if we had an oracle that tells us ")
        f.write("which retriever (BM25, BGE, or ELSER) to use for each query/turn.\n\n")
        f.write(f"**Oracle Selection Metric**: {METRICS[PRIMARY_METRIC]} ({PRIMARY_METRIC})\n\n")
        f.write("---\n\n")
        
        # Overall Performance Comparison
        f.write("## Overall Performance Comparison\n\n")
        f.write("| Metric | BM25 | BGE | ELSER | Oracle | Oracle Gain vs Best |\n")
        f.write("|--------|------|-----|-------|--------|---------------------|\n")
        
        for metric_key, metric_name in METRICS.items():
            bm25_val = bm25_perf.get(metric_key, 0.0)
            bge_val = bge_perf.get(metric_key, 0.0)
            elser_val = elser_perf.get(metric_key, 0.0)
            oracle_val = oracle_perf.get(metric_key, 0.0)
            
            best_individual = max(bm25_val, bge_val, elser_val)
            gain = oracle_val - best_individual
            gain_pct = (gain / best_individual * 100) if best_individual > 0 else 0.0
            
            f.write(f"| {metric_name} | {bm25_val:.4f} | {bge_val:.4f} | {elser_val:.4f} | ")
            f.write(f"**{oracle_val:.4f}** | +{gain:.4f} (+{gain_pct:.2f}%) |\n")
        
        f.write("\n")
        
        # Oracle Choices Distribution
        f.write("## Oracle Choices Distribution\n\n")
        total_choices = sum(oracle_choices.values())
        f.write("| Retriever | Times Chosen | Percentage |\n")
        f.write("|-----------|--------------|------------|\n")
        for retriever in RETRIEVAL_METHODS:
            count = oracle_choices.get(retriever, 0)
            pct = (count / total_choices * 100) if total_choices > 0 else 0.0
            f.write(f"| {retriever.upper()} | {count} | {pct:.2f}% |\n")
        f.write("\n")
        
        # Domain Analysis
        f.write("## Performance by Domain\n\n")
        for domain in DOMAINS:
            if domain not in domain_analysis:
                continue
            
            domain_data = domain_analysis[domain]
            f.write(f"### {domain.upper()}\n\n")
            
            # Performance table
            f.write("| Metric | BM25 | BGE | ELSER | Oracle | Oracle Gain |\n")
            f.write("|--------|------|-----|-------|--------|-------------|\n")
            
            for metric_key, metric_name in METRICS.items():
                bm25_val = domain_data['bm25'].get(metric_key, 0.0)
                bge_val = domain_data['bge'].get(metric_key, 0.0)
                elser_val = domain_data['elser'].get(metric_key, 0.0)
                oracle_val = domain_data['oracle'].get(metric_key, 0.0)
                
                best_individual = max(bm25_val, bge_val, elser_val)
                gain = oracle_val - best_individual
                
                f.write(f"| {metric_name} | {bm25_val:.4f} | {bge_val:.4f} | {elser_val:.4f} | ")
                f.write(f"**{oracle_val:.4f}** | +{gain:.4f} |\n")
            
            # Oracle choices for this domain
            if 'oracle_choices' in domain_data:
                f.write("\n**Oracle Choices**: ")
                choices = domain_data['oracle_choices']
                total = sum(choices.values())
                choice_strs = [f"{ret.upper()}: {count} ({count/total*100:.1f}%)" 
                              for ret, count in choices.items()]
                f.write(", ".join(choice_strs))
                f.write("\n\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        
        # Find which retriever wins most often
        best_retriever = max(oracle_choices.items(), key=lambda x: x[1])[0]
        best_retriever_pct = (oracle_choices[best_retriever] / total_choices * 100) if total_choices > 0 else 0
        
        f.write(f"1. **Most Frequently Chosen**: {best_retriever.upper()} is chosen by the oracle ")
        f.write(f"{oracle_choices[best_retriever]} times ({best_retriever_pct:.2f}% of queries)\n\n")
        
        # Calculate average improvement
        improvements = []
        for metric_key in METRICS.keys():
            best_individual = max(
                bm25_perf.get(metric_key, 0.0),
                bge_perf.get(metric_key, 0.0),
                elser_perf.get(metric_key, 0.0)
            )
            oracle_val = oracle_perf.get(metric_key, 0.0)
            if best_individual > 0:
                improvements.append((oracle_val - best_individual) / best_individual * 100)
        
        avg_improvement = statistics.mean(improvements) if improvements else 0.0
        f.write(f"2. **Average Improvement**: Oracle provides an average improvement of ")
        f.write(f"{avg_improvement:.2f}% over the best individual retriever across all metrics\n\n")
        
        # Primary metric improvement
        best_primary = max(
            bm25_perf.get(PRIMARY_METRIC, 0.0),
            bge_perf.get(PRIMARY_METRIC, 0.0),
            elser_perf.get(PRIMARY_METRIC, 0.0)
        )
        oracle_primary = oracle_perf.get(PRIMARY_METRIC, 0.0)
        primary_improvement = ((oracle_primary - best_primary) / best_primary * 100) if best_primary > 0 else 0.0
        f.write(f"3. **Primary Metric ({METRICS[PRIMARY_METRIC]}) Improvement**: ")
        f.write(f"{primary_improvement:.2f}% improvement over best individual retriever ")
        f.write(f"({best_primary:.4f} → {oracle_primary:.4f})\n\n")
        
        f.write("---\n\n")
        f.write("*Generated by oracle_retriever_analysis.py*\n")


def main():
    """Main analysis function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define result directories
    bm25_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / "bm25" / "results"
    bge_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / "bge" / "results"
    elser_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / "elser" / "results"
    
    output_dir = project_root / "knowledgebase" / "retrieval" / "mono-t5-reranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each strategy
    for strategy in STRATEGIES:
        print(f"\nAnalyzing strategy: {strategy}")
        
        # Load results for all retrievers
        print("  Loading BM25 results...")
        bm25_results, bm25_domains = load_retrieval_results(bm25_dir, 'bm25', strategy)
        print(f"    Loaded {len(bm25_results)} tasks")
        
        print("  Loading BGE results...")
        bge_results, bge_domains = load_retrieval_results(bge_dir, 'bge', strategy)
        print(f"    Loaded {len(bge_results)} tasks")
        
        print("  Loading ELSER results...")
        elser_results, elser_domains = load_retrieval_results(elser_dir, 'elser', strategy)
        print(f"    Loaded {len(elser_results)} tasks")
        
        # Calculate individual performances
        print("  Calculating individual performances...")
        bm25_perf = calculate_individual_performance(bm25_results)
        bge_perf = calculate_individual_performance(bge_results)
        elser_perf = calculate_individual_performance(elser_results)
        
        # Calculate oracle performance
        print("  Calculating oracle performance...")
        oracle_perf, oracle_choices, _ = calculate_oracle_performance(
            bm25_results, bge_results, elser_results, PRIMARY_METRIC
        )
        
        print(f"    Oracle choices: {oracle_choices}")
        
        # Domain analysis
        print("  Analyzing by domain...")
        domain_analysis = analyze_by_domain(
            bm25_results, bge_results, elser_results,
            bm25_domains, bge_domains, elser_domains,
            PRIMARY_METRIC
        )
        
        # Generate report
        output_file = output_dir / f"oracle_analysis_{strategy}.md"
        print(f"  Generating report: {output_file}")
        generate_markdown_report(
            strategy, bm25_perf, bge_perf, elser_perf,
            oracle_perf, oracle_choices, domain_analysis, output_file
        )
        
        print(f"  ✓ Completed {strategy}")
    
    # Generate combined summary
    print("\nGenerating combined summary...")
    generate_combined_summary(output_dir)
    print("✓ Analysis complete!")


def generate_combined_summary(output_dir: Path):
    """Generate a combined summary across all strategies."""
    summary_file = output_dir / "oracle_analysis_summary.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Oracle Retriever Analysis: Combined Summary\n\n")
        f.write("## Overview\n\n")
        f.write("This document provides a combined summary of oracle retriever analysis ")
        f.write("across all query strategies (Last Turn, Query Rewrite, Full Questions).\n\n")
        f.write("The oracle analysis answers: **What if we could always pick the best ")
        f.write("retriever (BM25, BGE, or ELSER) for each query?**\n\n")
        f.write(f"**Oracle Selection Metric**: {METRICS[PRIMARY_METRIC]} ({PRIMARY_METRIC})\n\n")
        f.write("---\n\n")
        
        f.write("## Strategy Comparison\n\n")
        f.write("See individual strategy reports for detailed analysis:\n\n")
        for strategy in STRATEGIES:
            f.write(f"- [{strategy.upper()} Strategy](oracle_analysis_{strategy}.md)\n")
        
        f.write("\n---\n\n")
        f.write("## Key Questions Answered\n\n")
        f.write("1. **How much improvement can we get from oracle selection?**\n")
        f.write("   - See 'Oracle Gain vs Best' columns in individual reports\n\n")
        f.write("2. **Which retriever is chosen most often by the oracle?**\n")
        f.write("   - See 'Oracle Choices Distribution' sections\n\n")
        f.write("3. **Does oracle performance vary by domain?**\n")
        f.write("   - See 'Performance by Domain' sections\n\n")
        f.write("4. **Is oracle selection worth pursuing?**\n")
        f.write("   - If oracle provides significant gains, it suggests that ")
        f.write("retriever selection/ensemble methods could be valuable\n\n")
        f.write("---\n\n")
        f.write("*Generated by oracle_retriever_analysis.py*\n")


if __name__ == "__main__":
    main()
