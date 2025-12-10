"""
Utility functions for mono-T5 oracle selection analysis.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

# Constants
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

# DOMAINS = ['clapnq', 'fiqa', 'govt', 'cloud']
# RETRIEVAL_METHODS = ['bm25', 'bge', 'elser']
# QUERY_STRATEGIES = ['lastturn', 'rewrite', 'questions']

# Single domain testing configuration
DOMAINS = ['clapnq']
RETRIEVAL_METHODS = ['elser']  # Retrieval methods to use
QUERY_STRATEGIES = ['lastturn', 'rewrite', 'questions']  # Query strategies to compare


def load_retrieval_results_with_texts(
    results_dir: Path,
    retrieval_method: str,
    strategy: str,
    domains: List[str] = None
) -> Dict[str, Dict]:
    """
    Load retrieval results including document texts.
    
    Args:
        results_dir: Directory containing retrieval result files
        retrieval_method: Name of retrieval method (bm25, bge, elser)
        strategy: Query strategy (rewrite, lastturn, questions)
        domains: List of domains to load (defaults to DOMAINS)
    
    Returns:
        Dictionary mapping task_id to {
            'contexts': List of {document_id, text, score},
            'domain': domain name,
            'retriever_scores': evaluation scores
        }
    """
    if domains is None:
        domains = DOMAINS
    
    results = {}
    
    for domain in domains:
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
                    if task_id:
                        results[task_id] = {
                            'contexts': data.get('contexts', []),
                            'domain': domain,
                            'retriever_scores': data.get('retriever_scores', {})
                        }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return results


def load_queries(strategy: str, project_root: Path, domains: List[str] = None) -> Dict[str, str]:
    """
    Load queries for all domains.
    
    Args:
        strategy: Query strategy (rewrite, lastturn, questions)
        project_root: Root directory of the project
        domains: List of domains to load (defaults to DOMAINS)
    
    Returns:
        Dictionary mapping query_id to query text
    """
    if domains is None:
        domains = DOMAINS
    
    queries = {}
    queries_dir = project_root / "human" / "retrieval_tasks"
    
    for domain in domains:
        query_file = queries_dir / domain / f"{domain}_{strategy}.jsonl"
        if not query_file.exists():
            print(f"Warning: {query_file} not found")
            continue
        
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    query_id = data.get('_id')
                    query_text = data.get('text', '')
                    if query_id:
                        queries[query_id] = query_text
        except Exception as e:
            print(f"Error loading {query_file}: {e}")
            continue
    
    return queries


def calculate_predicted_recall_at_k(
    documents: List[Dict],
    scores: List[float],
    k: int = 10,
    threshold: float = 0.5
) -> float:
    """
    Calculate predicted recall@k based on mono-T5 scores.
    
    We treat documents with mono-T5 score > threshold as "relevant".
    
    Args:
        documents: List of document dicts with document_id
        scores: List of mono-T5 relevance scores
        k: Top-k to consider
        threshold: Score threshold for relevance (default: 0.5)
    
    Returns:
        Predicted recall@k (fraction of top-k that are "relevant")
    """
    if len(documents) == 0:
        return 0.0
    
    # Sort by score (descending) and take top k
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_k = doc_score_pairs[:k]
    
    # Count how many are "relevant" (score > threshold)
    relevant_count = sum(1 for _, score in top_k if score > threshold)
    
    return relevant_count / len(top_k) if len(top_k) > 0 else 0.0


def calculate_performance_with_selection(
    all_results: Dict[str, Dict],
    selection_choices: Dict[str, str],
    metric: str = 'recall_10'
) -> Dict[str, float]:
    """
    Calculate aggregate performance when using selected retrievers.
    
    Args:
        all_results: Dict mapping retriever -> task_id -> results
        selection_choices: Dict mapping task_id -> selected retriever
        metric: Metric to aggregate
    
    Returns:
        Dictionary of metric -> average score
    """
    metric_scores = defaultdict(list)
    
    for task_id, selected_retriever in selection_choices.items():
        if selected_retriever not in all_results:
            continue
        
        retriever_results = all_results[selected_retriever]
        if task_id not in retriever_results:
            continue
        
        scores = retriever_results[task_id].get('retriever_scores', {})
        if metric in scores:
            metric_scores[metric].append(scores[metric])
        
        # Also collect all metrics for comprehensive analysis
        for m in METRICS.keys():
            if m in scores:
                metric_scores[m].append(scores[m])
    
    # Calculate averages
    avg_scores = {}
    for m, scores in metric_scores.items():
        if scores:
            avg_scores[m] = statistics.mean(scores)
        else:
            avg_scores[m] = 0.0
    
    return avg_scores


def calculate_oracle_performance(
    bm25_results: Dict[str, Dict],
    bge_results: Dict[str, Dict],
    elser_results: Dict[str, Dict],
    metric: str
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[float]]]:
    """
    Calculate oracle performance by always picking the best retriever for each task.
    
    Args:
        bm25_results: Dict mapping task_id -> retriever_scores
        bge_results: Dict mapping task_id -> retriever_scores
        elser_results: Dict mapping task_id -> retriever_scores
        metric: Primary metric to use for oracle selection
    
    Returns:
        Tuple of (oracle_avg_scores, oracle_choices, oracle_scores_by_metric)
    """
    all_task_ids = set()
    all_task_ids.update(bm25_results.keys())
    all_task_ids.update(bge_results.keys())
    all_task_ids.update(elser_results.keys())
    
    oracle_choices = defaultdict(int)
    oracle_scores_by_metric = defaultdict(list)
    
    for task_id in all_task_ids:
        bm25_scores = bm25_results.get(task_id, {})
        bge_scores = bge_results.get(task_id, {})
        elser_scores = elser_results.get(task_id, {})
        
        # Find oracle choice for primary metric
        scores = {}
        if bm25_scores and metric in bm25_scores:
            scores['bm25'] = bm25_scores[metric]
        if bge_scores and metric in bge_scores:
            scores['bge'] = bge_scores[metric]
        if elser_scores and metric in elser_scores:
            scores['elser'] = elser_scores[metric]
        
        if not scores:
            continue
        
        best_retriever = max(scores.items(), key=lambda x: x[1])[0]
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
    
    # Calculate averages
    oracle_avg_scores = {}
    for m, scores in oracle_scores_by_metric.items():
        if scores:
            oracle_avg_scores[m] = statistics.mean(scores)
        else:
            oracle_avg_scores[m] = 0.0
    
    return oracle_avg_scores, dict(oracle_choices), {}


def generate_report(
    individual_perfs: Dict[str, Dict[str, float]],
    monot5_perf: Dict[str, float],
    oracle_perf: Dict[str, float],
    monot5_choices: Dict[str, str],
    oracle_choices: Dict[str, int],
    output_file: Path
):
    """
    Generate markdown report comparing mono-T5 selection to individual retrievers and oracle.
    
    Args:
        individual_perfs: Dict mapping retriever -> metric -> score
        monot5_perf: Dict mapping metric -> score for mono-T5 selection
        oracle_perf: Dict mapping metric -> score for oracle selection
        monot5_choices: Dict mapping task_id -> selected retriever (mono-T5)
        oracle_choices: Dict mapping retriever -> count (oracle)
        output_file: Path to output markdown file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Mono-T5 Oracle Selection Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis uses **mono-T5** to predict which retriever (BM25, BGE, or ELSER) ")
        f.write("performs best for each query, then compares this selection to the oracle (ground truth).\n\n")
        f.write("**Methodology**:\n")
        f.write("1. For each query, retrieve documents using all three retrievers\n")
        f.write("2. Use mono-T5 to score each document-query pair\n")
        f.write("3. Calculate predicted recall@10 based on mono-T5 scores (treating score > 0.5 as relevant)\n")
        f.write("4. Select the retriever with the highest predicted recall@10\n")
        f.write("5. Evaluate performance using ground truth qrels\n\n")
        f.write("---\n\n")
        
        # Performance Comparison
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | BM25 | BGE | ELSER | Mono-T5 Selection | Oracle |\n")
        f.write("|--------|------|-----|-------|-------------------|--------|\n")
        
        for metric_key, metric_name in METRICS.items():
            bm25_val = individual_perfs['bm25'].get(metric_key, 0.0)
            bge_val = individual_perfs['bge'].get(metric_key, 0.0)
            elser_val = individual_perfs['elser'].get(metric_key, 0.0)
            monot5_val = monot5_perf.get(metric_key, 0.0)
            oracle_val = oracle_perf.get(metric_key, 0.0)
            
            f.write(f"| {metric_name} | {bm25_val:.4f} | {bge_val:.4f} | {elser_val:.4f} | ")
            f.write(f"**{monot5_val:.4f}** | {oracle_val:.4f} |\n")
        
        f.write("\n")
        
        # Selection Distribution
        f.write("## Selection Distribution\n\n")
        monot5_dist = Counter(monot5_choices.values())
        total = sum(monot5_dist.values())
        
        f.write("### Mono-T5 Selection\n\n")
        f.write("| Retriever | Times Chosen | Percentage |\n")
        f.write("|-----------|--------------|------------|\n")
        for retriever in RETRIEVAL_METHODS:
            count = monot5_dist.get(retriever, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            f.write(f"| {retriever.upper()} | {count} | {pct:.2f}% |\n")
        
        f.write("\n### Oracle Selection (Ground Truth)\n\n")
        f.write("| Retriever | Times Chosen | Percentage |\n")
        f.write("|-----------|--------------|------------|\n")
        oracle_total = sum(oracle_choices.values())
        for retriever in RETRIEVAL_METHODS:
            count = oracle_choices.get(retriever, 0)
            pct = (count / oracle_total * 100) if oracle_total > 0 else 0.0
            f.write(f"| {retriever.upper()} | {count} | {pct:.2f}% |\n")
        
        f.write("\n")
        
        # Key Metrics
        f.write("## Key Metrics\n\n")
        best_individual = max(
            individual_perfs['bm25'].get('recall_10', 0.0),
            individual_perfs['bge'].get('recall_10', 0.0),
            individual_perfs['elser'].get('recall_10', 0.0)
        )
        monot5_recall = monot5_perf.get('recall_10', 0.0)
        oracle_recall = oracle_perf.get('recall_10', 0.0)
        
        improvement_over_best = ((monot5_recall - best_individual) / best_individual * 100) if best_individual > 0 else 0.0
        gap_to_oracle = oracle_recall - monot5_recall
        oracle_gap_pct = (gap_to_oracle / oracle_recall * 100) if oracle_recall > 0 else 0.0
        
        f.write(f"- **Mono-T5 Recall@10**: {monot5_recall:.4f}\n")
        f.write(f"- **Best Individual Retriever Recall@10**: {best_individual:.4f}\n")
        f.write(f"- **Improvement over Best**: {improvement_over_best:.2f}%\n")
        f.write(f"- **Oracle Recall@10**: {oracle_recall:.4f}\n")
        f.write(f"- **Gap to Oracle**: {gap_to_oracle:.4f} ({oracle_gap_pct:.2f}%)\n")
        
        f.write("\n---\n\n")
        f.write("*Generated by mono_t5_oracle_selection.py*\n")


def save_results_as_json(
    monot5_choices: Dict[str, str],
    predicted_recalls: Dict[str, Dict[str, float]],
    individual_perfs: Dict[str, Dict[str, float]],
    monot5_perf: Dict[str, float],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int],
    output_dir: Path
):
    """
    Save analysis results as JSON files.
    
    Args:
        monot5_choices: Dict mapping task_id -> selected retriever
        predicted_recalls: Dict mapping task_id -> retriever -> predicted recall
        individual_perfs: Dict mapping retriever -> metric -> score
        monot5_perf: Dict mapping metric -> score for mono-T5 selection
        oracle_perf: Dict mapping metric -> score for oracle selection
        oracle_choices: Dict mapping retriever -> count (oracle)
        output_dir: Directory to save JSON files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save mono-T5 choices (task-level selections)
    choices_file = output_dir / "monot5_choices.json"
    with open(choices_file, 'w', encoding='utf-8') as f:
        json.dump(monot5_choices, f, indent=2, ensure_ascii=False)
    print(f"  Saved mono-T5 choices to {choices_file}")
    
    # 2. Save predicted recalls per task
    predicted_recalls_file = output_dir / "predicted_recalls.json"
    # Convert defaultdict to regular dict for JSON serialization
    predicted_recalls_dict = {k: dict(v) for k, v in predicted_recalls.items()}
    with open(predicted_recalls_file, 'w', encoding='utf-8') as f:
        json.dump(predicted_recalls_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved predicted recalls to {predicted_recalls_file}")
    
    # 3. Save performance metrics
    performance_file = output_dir / "performance_metrics.json"
    performance_data = {
        'individual_retrievers': individual_perfs,
        'monot5_selection': monot5_perf,
        'oracle': oracle_perf
    }
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved performance metrics to {performance_file}")
    
    # 4. Save selection distributions
    distribution_file = output_dir / "selection_distributions.json"
    monot5_dist = Counter(monot5_choices.values())
    distribution_data = {
        'monot5_selection': {
            retriever: {
                'count': monot5_dist.get(retriever, 0),
                'percentage': (monot5_dist.get(retriever, 0) / sum(monot5_dist.values()) * 100) 
                              if sum(monot5_dist.values()) > 0 else 0.0
            }
            for retriever in RETRIEVAL_METHODS
        },
        'oracle_selection': {
            retriever: {
                'count': oracle_choices.get(retriever, 0),
                'percentage': (oracle_choices.get(retriever, 0) / sum(oracle_choices.values()) * 100)
                              if sum(oracle_choices.values()) > 0 else 0.0
            }
            for retriever in RETRIEVAL_METHODS
        }
    }
    with open(distribution_file, 'w', encoding='utf-8') as f:
        json.dump(distribution_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved selection distributions to {distribution_file}")
    
    # 5. Save summary statistics
    summary_file = output_dir / "summary.json"
    best_individual = max(
        individual_perfs['bm25'].get('recall_10', 0.0),
        individual_perfs['bge'].get('recall_10', 0.0),
        individual_perfs['elser'].get('recall_10', 0.0)
    )
    monot5_recall = monot5_perf.get('recall_10', 0.0)
    oracle_recall = oracle_perf.get('recall_10', 0.0)
    
    summary_data = {
        'monot5_recall_10': monot5_recall,
        'best_individual_recall_10': best_individual,
        'improvement_over_best_percent': ((monot5_recall - best_individual) / best_individual * 100) 
                                         if best_individual > 0 else 0.0,
        'oracle_recall_10': oracle_recall,
        'gap_to_oracle': oracle_recall - monot5_recall,
        'gap_to_oracle_percent': ((oracle_recall - monot5_recall) / oracle_recall * 100) 
                                 if oracle_recall > 0 else 0.0,
        'total_tasks': len(monot5_choices),
        'strategy': STRATEGY
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved summary to {summary_file}")


def calculate_oracle_performance_strategies(
    lastturn_results: Dict[str, Dict],
    rewrite_results: Dict[str, Dict],
    questions_results: Dict[str, Dict],
    metric: str
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[float]]]:
    """
    Calculate oracle performance by always picking the best query strategy for each task.
    
    Args:
        lastturn_results: Dict mapping task_id -> retriever_scores
        rewrite_results: Dict mapping task_id -> retriever_scores
        questions_results: Dict mapping task_id -> retriever_scores
        metric: Primary metric to use for oracle selection
    
    Returns:
        Tuple of (oracle_avg_scores, oracle_choices, oracle_scores_by_metric)
    """
    all_task_ids = set()
    all_task_ids.update(lastturn_results.keys())
    all_task_ids.update(rewrite_results.keys())
    all_task_ids.update(questions_results.keys())
    
    oracle_choices = defaultdict(int)
    oracle_scores_by_metric = defaultdict(list)
    
    for task_id in all_task_ids:
        lastturn_scores = lastturn_results.get(task_id, {})
        rewrite_scores = rewrite_results.get(task_id, {})
        questions_scores = questions_results.get(task_id, {})
        
        # Find oracle choice for primary metric
        scores = {}
        if lastturn_scores and metric in lastturn_scores:
            scores['lastturn'] = lastturn_scores[metric]
        if rewrite_scores and metric in rewrite_scores:
            scores['rewrite'] = rewrite_scores[metric]
        if questions_scores and metric in questions_scores:
            scores['questions'] = questions_scores[metric]
        
        if not scores:
            continue
        
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        oracle_choices[best_strategy] += 1
        
        # Get scores for all metrics using the oracle's choice
        chosen_scores = None
        if best_strategy == 'lastturn':
            chosen_scores = lastturn_scores
        elif best_strategy == 'rewrite':
            chosen_scores = rewrite_scores
        elif best_strategy == 'questions':
            chosen_scores = questions_scores
        
        if chosen_scores:
            for m in METRICS.keys():
                if m in chosen_scores:
                    oracle_scores_by_metric[m].append(chosen_scores[m])
    
    # Calculate averages
    oracle_avg_scores = {}
    for m, scores in oracle_scores_by_metric.items():
        if scores:
            oracle_avg_scores[m] = statistics.mean(scores)
        else:
            oracle_avg_scores[m] = 0.0
    
    return oracle_avg_scores, dict(oracle_choices), {}


def save_results_as_json_strategies(
    monot5_choices: Dict[str, str],
    predicted_recalls: Dict[str, Dict[str, float]],
    individual_perfs: Dict[str, Dict[str, float]],
    monot5_perf: Dict[str, float],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int],
    output_dir: Path
):
    """
    Save analysis results as JSON files (for strategy selection).
    
    Args:
        monot5_choices: Dict mapping task_id -> selected strategy
        predicted_recalls: Dict mapping task_id -> strategy -> predicted recall
        individual_perfs: Dict mapping strategy -> metric -> score
        monot5_perf: Dict mapping metric -> score for mono-T5 selection
        oracle_perf: Dict mapping metric -> score for oracle selection
        oracle_choices: Dict mapping strategy -> count (oracle)
        output_dir: Directory to save JSON files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save mono-T5 choices (task-level selections)
    choices_file = output_dir / "monot5_choices.json"
    with open(choices_file, 'w', encoding='utf-8') as f:
        json.dump(monot5_choices, f, indent=2, ensure_ascii=False)
    print(f"  Saved mono-T5 choices to {choices_file}")
    
    # 2. Save predicted recalls per task
    predicted_recalls_file = output_dir / "predicted_recalls.json"
    # Convert defaultdict to regular dict for JSON serialization
    predicted_recalls_dict = {k: dict(v) for k, v in predicted_recalls.items()}
    with open(predicted_recalls_file, 'w', encoding='utf-8') as f:
        json.dump(predicted_recalls_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved predicted recalls to {predicted_recalls_file}")
    
    # 3. Save performance metrics
    performance_file = output_dir / "performance_metrics.json"
    performance_data = {
        'individual_strategies': individual_perfs,
        'monot5_selection': monot5_perf,
        'oracle': oracle_perf
    }
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved performance metrics to {performance_file}")
    
    # 4. Save selection distributions
    distribution_file = output_dir / "selection_distributions.json"
    monot5_dist = Counter(monot5_choices.values())
    distribution_data = {
        'monot5_selection': {
            strategy: {
                'count': monot5_dist.get(strategy, 0),
                'percentage': (monot5_dist.get(strategy, 0) / sum(monot5_dist.values()) * 100) 
                              if sum(monot5_dist.values()) > 0 else 0.0
            }
            for strategy in QUERY_STRATEGIES
        },
        'oracle_selection': {
            strategy: {
                'count': oracle_choices.get(strategy, 0),
                'percentage': (oracle_choices.get(strategy, 0) / sum(oracle_choices.values()) * 100)
                              if sum(oracle_choices.values()) > 0 else 0.0
            }
            for strategy in QUERY_STRATEGIES
        }
    }
    with open(distribution_file, 'w', encoding='utf-8') as f:
        json.dump(distribution_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved selection distributions to {distribution_file}")
    
    # 5. Save summary statistics
    summary_file = output_dir / "summary.json"
    best_individual = max(
        individual_perfs.get('lastturn', {}).get('recall_10', 0.0),
        individual_perfs.get('rewrite', {}).get('recall_10', 0.0),
        individual_perfs.get('questions', {}).get('recall_10', 0.0)
    )
    monot5_recall = monot5_perf.get('recall_10', 0.0)
    oracle_recall = oracle_perf.get('recall_10', 0.0)
    
    summary_data = {
        'monot5_recall_10': monot5_recall,
        'best_individual_recall_10': best_individual,
        'improvement_over_best_percent': ((monot5_recall - best_individual) / best_individual * 100) 
                                         if best_individual > 0 else 0.0,
        'oracle_recall_10': oracle_recall,
        'gap_to_oracle': oracle_recall - monot5_recall,
        'gap_to_oracle_percent': ((oracle_recall - monot5_recall) / oracle_recall * 100) 
                                 if oracle_recall > 0 else 0.0,
        'total_tasks': len(monot5_choices),
        'retrieval_methods': RETRIEVAL_METHODS,
        'query_strategies': QUERY_STRATEGIES
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved summary to {summary_file}")
