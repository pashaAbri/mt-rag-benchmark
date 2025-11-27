#!/usr/bin/env python3
"""
Analyze ELSER relevance scores by enrichment subtypes across different query strategies.

This script:
1. Loads enrichment data from cleaned_data/tasks/
2. Loads retrieval results from scripts/baselines/retrieval_scripts/elser/results/
3. Matches tasks with retrieval results by task_id
4. Calculates relevance score statistics (top-1 score) for each enrichment subtype
5. Compares relevance scores across strategies (lastturn, rewrite, questions)
"""

import json
import csv
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

STRATEGIES = ['lastturn', 'rewrite', 'questions']
DOMAINS = ['clapnq', 'fiqa', 'govt', 'cloud']
RETRIEVAL_METHOD = 'elser'  # Relevance scores are specific to ELSER


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


def load_relevance_scores(results_dir: Path, strategy: str) -> Dict[str, float]:
    """
    Load top-1 relevance scores for a specific strategy.
    
    Args:
        results_dir: Path to results directory
        strategy: One of 'lastturn', 'rewrite', 'questions'
    
    Returns:
        Dictionary mapping task_id to top-1 relevance score
    """
    scores = {}
    
    # Load domain-specific files
    # Note: We check for both individual domain files and the 'all' file if needed, 
    # but here we follow the pattern of iterating domains.
    for domain in DOMAINS:
        filename = f"{RETRIEVAL_METHOD}_{domain}_{strategy}_evaluated.jsonl"
        filepath = results_dir / filename
        
        if not filepath.exists():
            # Try checking if there are unevaluated files which also contain the scores
            filename = f"{RETRIEVAL_METHOD}_{domain}_{strategy}.jsonl"
            filepath = results_dir / filename
            
            if not filepath.exists():
                # Fallback to 'all' file if domain files missing? 
                # Ideally we prefer specific domain files to ensure we get what we expect.
                # For now, we just log a warning.
                # print(f"Warning: {filename} not found")
                continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    task_id = data.get('task_id')
                    
                    # Extract top-1 score
                    if task_id and 'contexts' in data and len(data['contexts']) > 0:
                        top_score = data['contexts'][0].get('score')
                        if top_score is not None:
                            scores[task_id] = float(top_score)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    # If domain files were missing, try to load from the 'all' file as fallback
    if not scores:
        filename = f"{RETRIEVAL_METHOD}_all_{strategy}_evaluated.jsonl"
        filepath = results_dir / filename
        if not filepath.exists():
             filename = f"{RETRIEVAL_METHOD}_all_{strategy}.jsonl"
             filepath = results_dir / filename
        
        if filepath.exists():
            print(f"  Loading from aggregate file: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        task_id = data.get('task_id')
                        if task_id and 'contexts' in data and len(data['contexts']) > 0:
                            top_score = data['contexts'][0].get('score')
                            if top_score is not None:
                                scores[task_id] = float(top_score)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    return scores


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
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def analyze_relevance_scores(
    enrichments: Dict[str, Dict],
    relevance_scores: Dict[str, Dict[str, float]]
) -> Tuple[Dict, Dict[str, int], Dict]:
    """
    Analyze relevance scores by enrichment subtype and domain.
    
    Args:
        enrichments: Task enrichments dict
        relevance_scores: Dict mapping strategy -> task_id -> score
    
    Returns:
        Tuple of (analysis dict, total_counts dict, domain_analysis dict)
        - analysis: Nested dictionary with statistics (aggregated across domains)
        - total_counts: Dictionary mapping enrichment_type -> total count
        - domain_analysis: Nested dictionary with statistics by domain
    """
    # Structure: enrichment_type -> subtype -> strategy -> stats (aggregated)
    analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # Structure: enrichment_type -> subtype -> domain -> strategy -> stats (by domain)
    domain_analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    # Track total counts per enrichment type for percentage calculation
    total_counts = defaultdict(int)
    
    # Group tasks by enrichment subtypes
    for task_id, task_data in enrichments.items():
        task_enrichments = task_data['enrichments']
        domain = task_data.get('domain', 'unknown')
        subtypes = extract_enrichment_subtypes(task_enrichments)
        
        # Check if task has scores for any strategy
        has_scores = any(
            strategy in relevance_scores and task_id in relevance_scores[strategy]
            for strategy in STRATEGIES
        )
        
        if not has_scores:
            continue
        
        # For each enrichment type
        for enrichment_type, subtype_list in subtypes.items():
            if not subtype_list:
                continue
            
            # For each subtype (a task can have multiple question types)
            for subtype in subtype_list:
                # For each strategy
                for strategy in STRATEGIES:
                    if strategy not in relevance_scores:
                        continue
                    if task_id not in relevance_scores[strategy]:
                        continue
                    
                    score = relevance_scores[strategy][task_id]
                    
                    # Store value for aggregated statistics
                    if 'values' not in analysis[enrichment_type][subtype][strategy]:
                        analysis[enrichment_type][subtype][strategy]['values'] = []
                    analysis[enrichment_type][subtype][strategy]['values'].append(score)
                    
                    # Store value for domain-specific statistics
                    if 'values' not in domain_analysis[enrichment_type][subtype][domain][strategy]:
                        domain_analysis[enrichment_type][subtype][domain][strategy]['values'] = []
                    domain_analysis[enrichment_type][subtype][domain][strategy]['values'].append(score)
                
                # Count this subtype occurrence (count once per task, not per strategy)
                total_counts[enrichment_type] += 1
    
    # Calculate statistics for aggregated groups
    for enrichment_type in analysis:
        for subtype in analysis[enrichment_type]:
            for strategy in analysis[enrichment_type][subtype]:
                values = analysis[enrichment_type][subtype][strategy]['values']
                stats = calculate_statistics(values)
                analysis[enrichment_type][subtype][strategy] = stats
    
    # Calculate statistics for domain-specific groups
    for enrichment_type in domain_analysis:
        for subtype in domain_analysis[enrichment_type]:
            for domain in domain_analysis[enrichment_type][subtype]:
                for strategy in domain_analysis[enrichment_type][subtype][domain]:
                    values = domain_analysis[enrichment_type][subtype][domain][strategy]['values']
                    stats = calculate_statistics(values)
                    domain_analysis[enrichment_type][subtype][domain][strategy] = stats
    
    return dict(analysis), dict(total_counts), dict(domain_analysis)


def print_summary_table(analysis: Dict, enrichment_type: str, total_counts: Dict[str, int]):
    """Print a summary table comparing strategies for each subtype."""
    print(f"\n{'='*100}")
    print(f"{enrichment_type.upper()} - ELSER Relevance Scores (Mean)")
    print(f"{'='*100}\n")
    
    # Collect all subtypes
    subtypes = sorted(analysis[enrichment_type].keys())
    
    if not subtypes:
        print("No data available.")
        return
    
    # Get total count for this enrichment type
    total_count = total_counts.get(enrichment_type, 0)
    
    # Print header
    print(f"{'Subtype':<25} {'Count':>8} {'%':>7} ", end="")
    for strategy in STRATEGIES:
        print(f"{strategy.upper():>12} ", end="")
    print()
    print("-" * 90)
    
    # Print rows
    for subtype in subtypes:
        # Get count (use any strategy, they should be similar)
        count = 0
        for strategy in STRATEGIES:
            if strategy in analysis[enrichment_type][subtype]:
                count = analysis[enrichment_type][subtype][strategy]['count']
                break
        
        # Calculate percentage
        percentage = (count / total_count * 100) if total_count > 0 else 0.0
        
        print(f"{subtype:<25} {count:>8} {percentage:>6.1f}% ", end="")
        
        for strategy in STRATEGIES:
            if strategy in analysis[enrichment_type][subtype]:
                mean = analysis[enrichment_type][subtype][strategy]['mean']
                print(f"{mean:>12.2f} ", end="")
            else:
                print(f"{'N/A':>12} ", end="")
        print()
    
    print(f"Total: {total_count} tasks")
    print()


def save_results_to_csv(analysis: Dict, total_counts: Dict[str, int], domain_analysis: Dict, output_dir: Path):
    """Save analysis results to CSV files, including domain breakdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results with 'all' domain and individual domains
    for enrichment_type in domain_analysis:
        rows = []
        total_count = total_counts.get(enrichment_type, 0)
        
        # First, add 'all' domain rows (aggregated statistics)
        if enrichment_type in analysis:
            for subtype in analysis[enrichment_type]:
                # Get count for this subtype (use any strategy)
                subtype_count = 0
                for strategy in STRATEGIES:
                    if strategy in analysis[enrichment_type][subtype]:
                        subtype_count = analysis[enrichment_type][subtype][strategy]['count']
                        break
                
                # Calculate percentage
                percentage = (subtype_count / total_count * 100) if total_count > 0 else 0.0
                
                for strategy in STRATEGIES:
                    if strategy not in analysis[enrichment_type][subtype]:
                        continue
                    
                    stats = analysis[enrichment_type][subtype][strategy]
                    rows.append({
                        'enrichment_type': enrichment_type,
                        'subtype': subtype,
                        'domain': 'all',
                        'strategy': strategy,
                        'count': stats['count'],
                        'percentage': round(percentage, 2),
                        'mean_score': stats['mean'],
                        'median_score': stats['median'],
                        'std_dev': stats['std'],
                        'min_score': stats['min'],
                        'max_score': stats['max']
                    })
        
        # Calculate domain-specific total counts for percentage calculation
        domain_total_counts = defaultdict(int)
        for subtype in domain_analysis[enrichment_type]:
            for domain in domain_analysis[enrichment_type][subtype]:
                # Count tasks per domain (use any strategy)
                for strategy in STRATEGIES:
                    if strategy in domain_analysis[enrichment_type][subtype][domain]:
                        domain_total_counts[domain] += domain_analysis[enrichment_type][subtype][domain][strategy]['count']
                        break
        
        # Then, add individual domain rows
        for subtype in domain_analysis[enrichment_type]:
            for domain in sorted(domain_analysis[enrichment_type][subtype].keys()):
                # Get count for this subtype-domain combination
                subtype_domain_count = 0
                for strategy in STRATEGIES:
                    if strategy in domain_analysis[enrichment_type][subtype][domain]:
                        subtype_domain_count = domain_analysis[enrichment_type][subtype][domain][strategy]['count']
                        break
                
                # Calculate percentage within domain
                domain_total = domain_total_counts.get(domain, 0)
                percentage = (subtype_domain_count / domain_total * 100) if domain_total > 0 else 0.0
                
                for strategy in STRATEGIES:
                    if strategy not in domain_analysis[enrichment_type][subtype][domain]:
                        continue
                    
                    stats = domain_analysis[enrichment_type][subtype][domain][strategy]
                    rows.append({
                        'enrichment_type': enrichment_type,
                        'subtype': subtype,
                        'domain': domain,
                        'strategy': strategy,
                        'count': stats['count'],
                        'percentage': round(percentage, 2),
                        'mean_score': stats['mean'],
                        'median_score': stats['median'],
                        'std_dev': stats['std'],
                        'min_score': stats['min'],
                        'max_score': stats['max']
                    })
        
        if rows:
            filename = f"relevance_scores_{enrichment_type}.csv"
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
    print("ELSER Relevance Score Analysis")
    print("="*100)
    print(f"\nLoading task enrichments from: {tasks_dir}")
    
    # Load enrichments
    enrichments = load_task_enrichments(tasks_dir)
    print(f"Loaded {len(enrichments)} tasks with enrichments")
    
    # Load relevance scores for each strategy
    print("\nLoading ELSER relevance scores...")
    relevance_scores = {}
    
    for strategy in STRATEGIES:
        print(f"  Loading {strategy}...", end=" ")
        scores = load_relevance_scores(results_dir, strategy)
        relevance_scores[strategy] = scores
        print(f"{len(scores)} tasks")
    
    # Analyze performance
    print("\nAnalyzing relevance scores by enrichment subtype and domain...")
    analysis, total_counts, domain_analysis = analyze_relevance_scores(enrichments, relevance_scores)
    
    # Print summary tables
    print("\n" + "="*100)
    print("SUMMARY TABLES (Aggregated across domains)")
    print("="*100)
    
    # Question Type summary
    print_summary_table(analysis, 'question_types', total_counts)
    
    # Multi-Turn summary
    print_summary_table(analysis, 'multi_turn', total_counts)
    
    # Answerability summary
    print_summary_table(analysis, 'answerability', total_counts)
    
    # Save results to CSV
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    save_results_to_csv(analysis, total_counts, domain_analysis, output_dir)
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

