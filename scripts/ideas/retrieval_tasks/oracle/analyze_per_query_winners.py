"""
Analyze which strategy performs best for each query.

This script compares lastturn, rewrite, and questions strategies at the query level
to determine if routing (selecting the best strategy per query) would be beneficial.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_evaluated_results(filepath: str) -> Dict:
    """Load evaluated results with per-query scores."""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data['task_id']
            scores = data['retriever_scores']
            results[task_id] = {
                'recall_5': scores['recall_5'],
                'ndcg_5': scores['ndcg_cut_5']
            }
    return results


def compare_strategies(
    lastturn_results: Dict,
    rewrite_results: Dict,
    questions_results: Dict
) -> Dict:
    """
    Compare strategies per query and determine winners.
    
    Returns:
        {
            'lastturn_wins': count,
            'rewrite_wins': count,
            'questions_wins': count,
            'ties': count,
            'details': [(query_id, winner, recall_scores, ndcg_scores), ...]
        }
    """
    lastturn_wins = 0
    rewrite_wins = 0
    questions_wins = 0
    ties = 0
    
    details = []
    
    # Get common query IDs
    common_queries = set(lastturn_results.keys()) & set(rewrite_results.keys()) & set(questions_results.keys())
    
    for query_id in sorted(common_queries):
        lt_recall = lastturn_results[query_id]['recall_5']
        rw_recall = rewrite_results[query_id]['recall_5']
        q_recall = questions_results[query_id]['recall_5']
        
        lt_ndcg = lastturn_results[query_id]['ndcg_5']
        rw_ndcg = rewrite_results[query_id]['ndcg_5']
        q_ndcg = questions_results[query_id]['ndcg_5']
        
        # Determine winner based on Recall@5 (primary metric)
        scores = [
            ('lastturn', lt_recall, lt_ndcg),
            ('rewrite', rw_recall, rw_ndcg),
            ('questions', q_recall, q_ndcg)
        ]
        
        # Sort by recall (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Check for ties
        if scores[0][1] == scores[1][1] == scores[2][1]:
            winner = 'tie_3way'
            ties += 1
        elif scores[0][1] == scores[1][1]:
            winner = 'tie_2way'
            ties += 1
        else:
            winner = scores[0][0]
            if winner == 'lastturn':
                lastturn_wins += 1
            elif winner == 'rewrite':
                rewrite_wins += 1
            else:
                questions_wins += 1
        
        details.append((
            query_id,
            winner,
            {'lastturn': lt_recall, 'rewrite': rw_recall, 'questions': q_recall},
            {'lastturn': lt_ndcg, 'rewrite': rw_ndcg, 'questions': q_ndcg}
        ))
    
    return {
        'total_queries': len(common_queries),
        'lastturn_wins': lastturn_wins,
        'rewrite_wins': rewrite_wins,
        'questions_wins': questions_wins,
        'ties': ties,
        'details': details
    }


def calculate_oracle_performance(details: List[Tuple]) -> Tuple[float, float]:
    """
    Calculate upper-bound performance if we always picked the best strategy.
    
    Returns:
        (oracle_recall_avg, oracle_ndcg_avg)
    """
    total_recall = 0.0
    total_ndcg = 0.0
    
    for query_id, winner, recall_scores, ndcg_scores in details:
        # Pick the best recall
        best_recall = max(recall_scores.values())
        total_recall += best_recall
        
        # For that same strategy, get its nDCG
        best_strategy = max(recall_scores, key=recall_scores.get)
        total_ndcg += ndcg_scores[best_strategy]
    
    n = len(details)
    return (total_recall / n, total_ndcg / n)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze which strategy wins for each query'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        required=True,
        choices=['bge', 'bm25', 'elser'],
        help='Retriever to analyze'
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['clapnq', 'cloud', 'fiqa', 'govt'],
        help='Domain to analyze'
    )
    
    args = parser.parse_args()
    
    # Build file paths
    base_dir = Path('scripts/baselines/retrieval_scripts') / args.retriever / 'results'
    
    lastturn_file = base_dir / f'{args.retriever}_{args.domain}_lastturn_evaluated.jsonl'
    rewrite_file = base_dir / f'{args.retriever}_{args.domain}_rewrite_evaluated.jsonl'
    questions_file = base_dir / f'{args.retriever}_{args.domain}_questions_evaluated.jsonl'
    
    # Load results
    print(f"\n{'='*80}")
    print(f"Per-Query Strategy Analysis: {args.retriever.upper()} on {args.domain.upper()}")
    print(f"{'='*80}\n")
    
    print("Loading evaluated results...")
    lastturn = load_evaluated_results(lastturn_file)
    rewrite = load_evaluated_results(rewrite_file)
    questions = load_evaluated_results(questions_file)
    
    print(f"  Lastturn: {len(lastturn)} queries")
    print(f"  Rewrite: {len(rewrite)} queries")
    print(f"  Questions: {len(questions)} queries\n")
    
    # Compare strategies
    comparison = compare_strategies(lastturn, rewrite, questions)
    
    # Print summary
    print(f"{'='*80}")
    print(f"STRATEGY WINNER BREAKDOWN")
    print(f"{'='*80}\n")
    
    total = comparison['total_queries']
    print(f"Total queries analyzed: {total}\n")
    
    lt_pct = (comparison['lastturn_wins'] / total) * 100
    rw_pct = (comparison['rewrite_wins'] / total) * 100
    q_pct = (comparison['questions_wins'] / total) * 100
    tie_pct = (comparison['ties'] / total) * 100
    
    print(f"Lastturn wins:   {comparison['lastturn_wins']:3d} ({lt_pct:5.1f}%)")
    print(f"Rewrite wins:    {comparison['rewrite_wins']:3d} ({rw_pct:5.1f}%)")
    print(f"Questions wins:  {comparison['questions_wins']:3d} ({q_pct:5.1f}%)")
    print(f"Ties:            {comparison['ties']:3d} ({tie_pct:5.1f}%)\n")
    
    # Calculate actual performance of each strategy
    lastturn_avg_recall = sum(s['lastturn'] for _, _, s, _ in comparison['details']) / total
    rewrite_avg_recall = sum(s['rewrite'] for _, _, s, _ in comparison['details']) / total
    questions_avg_recall = sum(s['questions'] for _, _, s, _ in comparison['details']) / total
    
    lastturn_avg_ndcg = sum(s['lastturn'] for _, _, _, s in comparison['details']) / total
    rewrite_avg_ndcg = sum(s['rewrite'] for _, _, _, s in comparison['details']) / total
    questions_avg_ndcg = sum(s['questions'] for _, _, _, s in comparison['details']) / total
    
    print(f"{'='*80}")
    print(f"ACTUAL PERFORMANCE")
    print(f"{'='*80}\n")
    
    print(f"{'Strategy':<15} {'Recall@5':>12} {'nDCG@5':>12}")
    print(f"{'-'*40}")
    print(f"{'Lastturn':<15} {lastturn_avg_recall:>12.4f} {lastturn_avg_ndcg:>12.4f}")
    print(f"{'Rewrite':<15} {rewrite_avg_recall:>12.4f} {rewrite_avg_ndcg:>12.4f}")
    print(f"{'Questions':<15} {questions_avg_recall:>12.4f} {questions_avg_ndcg:>12.4f}\n")
    
    # Calculate oracle (upper bound) performance
    oracle_recall, oracle_ndcg = calculate_oracle_performance(comparison['details'])
    
    print(f"{'='*80}")
    print(f"ORACLE ROUTING (Perfect Strategy Selection)")
    print(f"{'='*80}\n")
    
    print(f"Oracle Recall@5: {oracle_recall:.4f}")
    print(f"Oracle nDCG@5:   {oracle_ndcg:.4f}\n")
    
    # Calculate potential improvement
    best_single_recall = max(lastturn_avg_recall, rewrite_avg_recall, questions_avg_recall)
    best_single_ndcg = max(lastturn_avg_ndcg, rewrite_avg_ndcg, questions_avg_ndcg)
    best_strategy_name = max(
        [('lastturn', lastturn_avg_recall), ('rewrite', rewrite_avg_recall), ('questions', questions_avg_recall)],
        key=lambda x: x[1]
    )[0]
    
    improvement_recall = ((oracle_recall - best_single_recall) / best_single_recall) * 100
    improvement_ndcg = ((oracle_ndcg - best_single_ndcg) / best_single_ndcg) * 100
    
    print(f"Best single strategy: {best_strategy_name.capitalize()}")
    print(f"  Recall@5: {best_single_recall:.4f}")
    print(f"  nDCG@5:   {best_single_ndcg:.4f}\n")
    
    print(f"Oracle improvement over best single strategy:")
    print(f"  Recall@5: +{improvement_recall:.1f}%")
    print(f"  nDCG@5:   +{improvement_ndcg:.1f}%\n")
    
    # Show some examples
    print(f"{'='*80}")
    print(f"EXAMPLE QUERIES WHERE STRATEGIES DIFFER")
    print(f"{'='*80}\n")
    
    # Find queries where lastturn beats rewrite
    lt_better = [d for d in comparison['details'] 
                 if d[2]['lastturn'] > d[2]['rewrite'] and d[2]['lastturn'] > 0][:3]
    
    if lt_better:
        print("Queries where LASTTURN beats REWRITE:")
        for query_id, winner, recall, ndcg in lt_better:
            print(f"  Query: {query_id}")
            print(f"    Lastturn R@5:  {recall['lastturn']:.2f}")
            print(f"    Rewrite R@5:   {recall['rewrite']:.2f}")
            print(f"    Questions R@5: {recall['questions']:.2f}\n")
    
    # Find queries where rewrite beats lastturn
    rw_better = [d for d in comparison['details'] 
                 if d[2]['rewrite'] > d[2]['lastturn'] and d[2]['rewrite'] > 0][:3]
    
    if rw_better:
        print("Queries where REWRITE beats LASTTURN:")
        for query_id, winner, recall, ndcg in rw_better:
            print(f"  Query: {query_id}")
            print(f"    Lastturn R@5:  {recall['lastturn']:.2f}")
            print(f"    Rewrite R@5:   {recall['rewrite']:.2f}")
            print(f"    Questions R@5: {recall['questions']:.2f}\n")
    
    # Find queries where questions beats both
    q_better = [d for d in comparison['details'] 
                if d[2]['questions'] > d[2]['lastturn'] and d[2]['questions'] > d[2]['rewrite']][:3]
    
    if q_better:
        print("Queries where QUESTIONS beats both:")
        for query_id, winner, recall, ndcg in q_better:
            print(f"  Query: {query_id}")
            print(f"    Lastturn R@5:  {recall['lastturn']:.2f}")
            print(f"    Rewrite R@5:   {recall['rewrite']:.2f}")
            print(f"    Questions R@5: {recall['questions']:.2f}\n")
    
    print(f"{'='*80}")
    print(f"ROUTING POTENTIAL")
    print(f"{'='*80}\n")
    
    if improvement_recall > 5.0:
        print("✅ HIGH POTENTIAL - Oracle routing could improve by >5%")
        print("   Worth exploring learned routing strategies!")
    elif improvement_recall > 2.0:
        print("⚠️ MODERATE POTENTIAL - Oracle routing could improve by 2-5%")
        print("   Routing may help, but requires cost/benefit analysis")
    else:
        print("❌ LOW POTENTIAL - Oracle routing improves by <2%")
        print("   One strategy dominates; routing not worth complexity")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

