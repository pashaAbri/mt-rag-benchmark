#!/usr/bin/env python3
"""
Analyze coverage of qrels by retrieval strategies.

For each task, checks if the union of retrieved documents from all three strategies
(lastturn, rewrite, questions) covers all the relevant documents in the qrels.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Domain names
DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Query strategies
STRATEGIES = ['lastturn', 'rewrite', 'questions']

# Retrieval method (assuming elser for now, but can be changed)
RETRIEVAL_METHOD = 'elser'

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'scripts' / 'baselines' / 'retrieval_scripts' / RETRIEVAL_METHOD / 'results'
QRELS_DIR = PROJECT_ROOT / 'human' / 'retrieval_tasks'


def load_qrels(domain: str) -> Dict[str, Set[str]]:
    """
    Load qrels for a domain.
    
    Returns:
        Dictionary mapping query_id to set of relevant document IDs
    """
    qrels_path = QRELS_DIR / domain / 'qrels' / 'dev.tsv'
    
    if not qrels_path.exists():
        print(f"Warning: Qrels file not found: {qrels_path}")
        return {}
    
    qrels = defaultdict(set)
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0], row[1], int(row[2])
            # Only consider documents with positive relevance scores
            if score > 0:
                qrels[query_id].add(doc_id)
    
    return dict(qrels)


def load_retrieval_results(domain: str, strategy: str) -> Dict[str, Set[str]]:
    """
    Load retrieval results for a domain and strategy.
    
    Returns:
        Dictionary mapping task_id to set of retrieved document IDs
    """
    filename = f"{RETRIEVAL_METHOD}_{domain}_{strategy}.jsonl"
    filepath = RESULTS_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: Retrieval results file not found: {filepath}")
        return {}
    
    results = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                contexts = data.get('contexts', [])
                
                if task_id:
                    retrieved_docs = set()
                    for ctx in contexts:
                        doc_id = ctx.get('document_id')
                        if doc_id:
                            retrieved_docs.add(doc_id)
                    results[task_id] = retrieved_docs
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {filepath}: {e}")
                continue
    
    return results


def analyze_coverage():
    """
    Analyze coverage of qrels by combined retrieval strategies.
    """
    all_stats = []
    
    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Analyzing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        # Load qrels
        qrels = load_qrels(domain)
        print(f"Loaded {len(qrels)} queries with qrels")
        
        # Load retrieval results for each strategy
        retrieval_results = {}
        for strategy in STRATEGIES:
            results = load_retrieval_results(domain, strategy)
            retrieval_results[strategy] = results
            print(f"  {strategy}: {len(results)} queries")
        
        # Find common task_ids across all strategies and qrels
        common_task_ids = set(qrels.keys())
        for strategy in STRATEGIES:
            if strategy in retrieval_results:
                common_task_ids &= set(retrieval_results[strategy].keys())
        
        print(f"\nCommon task_ids across all strategies and qrels: {len(common_task_ids)}")
        
        # Analyze coverage for each task
        coverage_stats = {
            'total_tasks': len(common_task_ids),
            'fully_covered': 0,
            'partially_covered': 0,
            'not_covered': 0,
            'total_qrel_docs': 0,
            'total_retrieved_docs': 0,
            'total_covered_docs': 0,
            'per_task_coverage': []
        }
        
        for task_id in common_task_ids:
            qrel_docs = qrels[task_id]
            
            # Get union of retrieved documents from all strategies
            combined_retrieved = set()
            for strategy in STRATEGIES:
                if strategy in retrieval_results and task_id in retrieval_results[strategy]:
                    combined_retrieved |= retrieval_results[strategy][task_id]
            
            # Calculate coverage
            covered_docs = qrel_docs & combined_retrieved
            coverage_ratio = len(covered_docs) / len(qrel_docs) if qrel_docs else 0.0
            
            coverage_stats['total_qrel_docs'] += len(qrel_docs)
            coverage_stats['total_retrieved_docs'] += len(combined_retrieved)
            coverage_stats['total_covered_docs'] += len(covered_docs)
            
            if coverage_ratio == 1.0:
                coverage_stats['fully_covered'] += 1
            elif coverage_ratio > 0:
                coverage_stats['partially_covered'] += 1
            else:
                coverage_stats['not_covered'] += 1
            
            coverage_stats['per_task_coverage'].append({
                'task_id': task_id,
                'qrel_count': len(qrel_docs),
                'retrieved_count': len(combined_retrieved),
                'covered_count': len(covered_docs),
                'coverage_ratio': coverage_ratio,
                'missing_docs': list(qrel_docs - covered_docs)
            })
        
        # Calculate aggregate statistics
        if coverage_stats['total_tasks'] > 0:
            avg_coverage = coverage_stats['total_covered_docs'] / coverage_stats['total_qrel_docs'] if coverage_stats['total_qrel_docs'] > 0 else 0.0
            coverage_stats['avg_coverage'] = avg_coverage
            coverage_stats['domain'] = domain
            
            all_stats.append(coverage_stats)
            
            # Print summary
            print(f"\nCoverage Summary for {domain}:")
            print(f"  Total tasks analyzed: {coverage_stats['total_tasks']}")
            print(f"  Fully covered (100%): {coverage_stats['fully_covered']} ({coverage_stats['fully_covered']/coverage_stats['total_tasks']*100:.1f}%)")
            print(f"  Partially covered: {coverage_stats['partially_covered']} ({coverage_stats['partially_covered']/coverage_stats['total_tasks']*100:.1f}%)")
            print(f"  Not covered (0%): {coverage_stats['not_covered']} ({coverage_stats['not_covered']/coverage_stats['total_tasks']*100:.1f}%)")
            print(f"\n  Total qrel documents: {coverage_stats['total_qrel_docs']}")
            print(f"  Total retrieved documents (union): {coverage_stats['total_retrieved_docs']}")
            print(f"  Total covered documents: {coverage_stats['total_covered_docs']}")
            print(f"  Average coverage: {avg_coverage*100:.2f}%")
            
            # Show tasks with missing coverage
            tasks_with_missing = [t for t in coverage_stats['per_task_coverage'] if t['coverage_ratio'] < 1.0]
            if tasks_with_missing:
                print(f"\n  Tasks with missing coverage: {len(tasks_with_missing)}")
                print(f"  Top 10 tasks with lowest coverage:")
                sorted_tasks = sorted(tasks_with_missing, key=lambda x: x['coverage_ratio'])
                for task in sorted_tasks[:10]:
                    print(f"    {task['task_id']}: {task['coverage_ratio']*100:.1f}% ({task['covered_count']}/{task['qrel_count']} docs)")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_tasks = sum(s['total_tasks'] for s in all_stats)
    total_fully_covered = sum(s['fully_covered'] for s in all_stats)
    total_partially_covered = sum(s['partially_covered'] for s in all_stats)
    total_not_covered = sum(s['not_covered'] for s in all_stats)
    total_qrel_docs = sum(s['total_qrel_docs'] for s in all_stats)
    total_covered_docs = sum(s['total_covered_docs'] for s in all_stats)
    
    print(f"Total tasks across all domains: {total_tasks}")
    print(f"Fully covered: {total_fully_covered} ({total_fully_covered/total_tasks*100:.1f}%)")
    print(f"Partially covered: {total_partially_covered} ({total_partially_covered/total_tasks*100:.1f}%)")
    print(f"Not covered: {total_not_covered} ({total_not_covered/total_tasks*100:.1f}%)")
    print(f"\nTotal qrel documents: {total_qrel_docs}")
    print(f"Total covered documents: {total_covered_docs}")
    print(f"Overall coverage: {total_covered_docs/total_qrel_docs*100:.2f}%")
    
    # Per-domain summary table
    print(f"\n{'='*60}")
    print("PER-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"{'Domain':<10} {'Tasks':<8} {'Fully':<8} {'Partial':<8} {'None':<8} {'Coverage':<10}")
    print("-" * 60)
    for stats in all_stats:
        domain = stats['domain']
        tasks = stats['total_tasks']
        fully = stats['fully_covered']
        partial = stats['partially_covered']
        none = stats['not_covered']
        coverage = stats.get('avg_coverage', 0.0) * 100
        print(f"{domain:<10} {tasks:<8} {fully:<8} {partial:<8} {none:<8} {coverage:<10.2f}%")
    
    # Save detailed results
    output_file = PROJECT_ROOT / 'scripts' / 'discovery' / 'qrel_coverage_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    analyze_coverage()
