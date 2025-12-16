#!/usr/bin/env python3
"""
Analyze per-query differences between targeted rewrite and standard rewrite.

This script:
1. Loads retrieval results and qrels
2. Computes per-query metrics using pytrec_eval
3. Identifies queries where targeted helps, hurts, or both miss
4. Correlates with analysis metadata (turn filtering, similarities)
5. Outputs examples for manual inspection

Usage:
    python analyze_per_query_diff.py
"""

import json
import csv
import pandas as pd
import pytrec_eval
from pathlib import Path
from collections import defaultdict

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

TARGETED_RESULTS_DIR = script_dir / "retrieval_results"
BASELINE_RESULTS_DIR = project_root / "scripts" / "baselines" / "retrieval_scripts" / "elser" / "results"
ANALYSIS_DIR = script_dir / "intermediate"
QRELS_DIR = project_root / "human" / "retrieval_tasks"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']


def load_qrels(domain):
    """Load qrels for a domain."""
    qrels_file = QRELS_DIR / domain / "qrels" / "dev.tsv"
    if not qrels_file.exists():
        return {}
    
    qrels = {}
    with open(qrels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0], row[1], int(row[2])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score
    return qrels


def load_retrieval_results(filepath):
    """Load retrieval results from JSONL file."""
    results = {}
    if not filepath.exists():
        return results
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                if task_id:
                    doc_scores = {}
                    for ctx in data.get('contexts', []):
                        doc_id = ctx.get('document_id')
                        score = ctx.get('score', 0.0)
                        if doc_id:
                            doc_scores[doc_id] = score
                    results[task_id] = doc_scores
            except json.JSONDecodeError:
                continue
    return results


def compute_per_query_metrics(qrels, results):
    """Compute per-query metrics using pytrec_eval."""
    if not qrels or not results:
        return {}
    
    # Filter results to only include queries in qrels
    filtered_results = {qid: docs for qid, docs in results.items() if qid in qrels}
    
    if not filtered_results:
        return {}
    
    k_values = [1, 3, 5, 10]
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, recall_string})
    scores = evaluator.evaluate(filtered_results)
    
    return scores


def load_analysis_data(domain):
    """Load analysis data with turn filtering info."""
    analysis_file = ANALYSIS_DIR / f"targeted_rewrite_{domain}_analysis.json"
    if not analysis_file.exists():
        return {}
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analyses = data.get("analyses", [])
        return {a['task_id']: a for a in analyses if 'task_id' in a}
    except Exception as e:
        print(f"Error loading analysis: {e}")
        return {}


def analyze_domain(domain):
    """Analyze per-query differences for a domain."""
    print(f"\n{'='*80}")
    print(f"DOMAIN: {domain.upper()}")
    print('='*80)
    
    # Load qrels
    qrels = load_qrels(domain)
    if not qrels:
        print(f"No qrels found for {domain}")
        return None
    print(f"Loaded {len(qrels)} qrels")
    
    # Load retrieval results
    targeted_file = TARGETED_RESULTS_DIR / f"targeted_rewrite_{domain}_elser.jsonl"
    baseline_file = BASELINE_RESULTS_DIR / f"elser_{domain}_rewrite.jsonl"
    
    targeted_results = load_retrieval_results(targeted_file)
    baseline_results = load_retrieval_results(baseline_file)
    
    if not targeted_results:
        print(f"No targeted results found: {targeted_file}")
        return None
    if not baseline_results:
        print(f"No baseline results found: {baseline_file}")
        return None
    
    print(f"Loaded {len(targeted_results)} targeted results, {len(baseline_results)} baseline results")
    
    # Compute per-query metrics
    targeted_scores = compute_per_query_metrics(qrels, targeted_results)
    baseline_scores = compute_per_query_metrics(qrels, baseline_results)
    
    print(f"Evaluated {len(targeted_scores)} targeted queries, {len(baseline_scores)} baseline queries")
    
    # Load analysis data
    analysis_data = load_analysis_data(domain)
    
    # Find common task_ids
    common_ids = set(targeted_scores.keys()) & set(baseline_scores.keys())
    print(f"Common queries for comparison: {len(common_ids)}")
    
    # Compute differences
    differences = []
    for task_id in common_ids:
        t = targeted_scores[task_id]
        b = baseline_scores[task_id]
        
        diff_ndcg_10 = t.get('ndcg_cut_10', 0) - b.get('ndcg_cut_10', 0)
        diff_recall_10 = t.get('recall_10', 0) - b.get('recall_10', 0)
        
        analysis = analysis_data.get(task_id, {})
        
        differences.append({
            'task_id': task_id,
            'baseline_ndcg_10': b.get('ndcg_cut_10', 0),
            'targeted_ndcg_10': t.get('ndcg_cut_10', 0),
            'diff_ndcg_10': diff_ndcg_10,
            'baseline_recall_10': b.get('recall_10', 0),
            'targeted_recall_10': t.get('recall_10', 0),
            'diff_recall_10': diff_recall_10,
            'num_history_turns': analysis.get('num_history_turns', 0),
            'selected_turns': analysis.get('selected_turns', 0),
            'above_threshold_count': analysis.get('above_threshold_count', 0),
            'turn_id': analysis.get('turn_id', 0),
            'original_query': analysis.get('original_query', ''),
            'rewritten_query': analysis.get('rewritten_query', ''),
        })
    
    df = pd.DataFrame(differences)
    
    # Summary statistics
    print(f"\n--- Summary Statistics ---")
    print(f"Mean diff nDCG@10: {df['diff_ndcg_10'].mean():.4f}")
    print(f"Median diff nDCG@10: {df['diff_ndcg_10'].median():.4f}")
    print(f"Std diff nDCG@10: {df['diff_ndcg_10'].std():.4f}")
    print(f"Min diff nDCG@10: {df['diff_ndcg_10'].min():.4f}")
    print(f"Max diff nDCG@10: {df['diff_ndcg_10'].max():.4f}")
    
    # Categorize queries
    helps = df[df['diff_ndcg_10'] > 0.05]  # Targeted helps significantly
    hurts = df[df['diff_ndcg_10'] < -0.05]  # Targeted hurts significantly
    neutral = df[(df['diff_ndcg_10'] >= -0.05) & (df['diff_ndcg_10'] <= 0.05)]
    both_miss = df[(df['baseline_ndcg_10'] < 0.3) & (df['targeted_ndcg_10'] < 0.3)]
    both_good = df[(df['baseline_ndcg_10'] >= 0.7) & (df['targeted_ndcg_10'] >= 0.7)]
    
    print(f"\n--- Categorization (threshold: ±0.05 nDCG@10) ---")
    print(f"Targeted HELPS: {len(helps)} ({100*len(helps)/len(df):.1f}%)")
    print(f"Targeted HURTS: {len(hurts)} ({100*len(hurts)/len(df):.1f}%)")
    print(f"NEUTRAL: {len(neutral)} ({100*len(neutral)/len(df):.1f}%)")
    print(f"Both MISS (<0.3): {len(both_miss)} ({100*len(both_miss)/len(df):.1f}%)")
    print(f"Both GOOD (≥0.7): {len(both_good)} ({100*len(both_good)/len(df):.1f}%)")
    
    # Analyze patterns
    print(f"\n--- Pattern Analysis ---")
    
    # By turn_id
    if 'turn_id' in df.columns and df['turn_id'].sum() > 0:
        by_turn = df.groupby('turn_id')['diff_ndcg_10'].agg(['mean', 'count', 'std'])
        print("\nBy Turn ID:")
        print(by_turn.to_string())
    
    # By number of turns filtered
    df['turns_filtered'] = df['num_history_turns'] - df['selected_turns']
    df['pct_filtered'] = df['turns_filtered'] / df['num_history_turns'].replace(0, 1)
    
    high_filter = df[df['pct_filtered'] > 0.5]
    low_filter = df[(df['pct_filtered'] <= 0.5) & (df['num_history_turns'] > 0)]
    no_history = df[df['num_history_turns'] == 0]
    
    print(f"\nBy filtering level:")
    print(f"  No history (turn 1):  n={len(no_history)}, mean diff={no_history['diff_ndcg_10'].mean():.4f}")
    print(f"  Low filtering (≤50%): n={len(low_filter)}, mean diff={low_filter['diff_ndcg_10'].mean():.4f}")
    print(f"  High filtering (>50%): n={len(high_filter)}, mean diff={high_filter['diff_ndcg_10'].mean():.4f}")
    
    # Correlation between filtering and improvement
    multi_turn = df[df['num_history_turns'] > 0]
    if len(multi_turn) > 5:
        corr = multi_turn['pct_filtered'].corr(multi_turn['diff_ndcg_10'])
        print(f"\nCorrelation (% filtered vs diff_ndcg_10): {corr:.3f}")
    
    return {
        'df': df,
        'helps': helps,
        'hurts': hurts,
        'neutral': neutral,
        'both_miss': both_miss,
    }


def show_examples(results, category, n=3):
    """Show example queries from a category."""
    df = results[category]
    if len(df) == 0:
        print(f"  No examples in {category}")
        return
    
    # Sort by magnitude of difference
    if category == 'helps':
        df = df.nlargest(n, 'diff_ndcg_10')
    elif category == 'hurts':
        df = df.nsmallest(n, 'diff_ndcg_10')
    elif category == 'both_miss':
        # Show ones with most history turns (more potential for filtering to help)
        df = df.nlargest(n, 'num_history_turns')
    
    for _, row in df.iterrows():
        print(f"\n  Task ID: {row['task_id']}")
        print(f"  Turn ID: {row['turn_id']}")
        print(f"  Baseline nDCG@10: {row['baseline_ndcg_10']:.3f}")
        print(f"  Targeted nDCG@10: {row['targeted_ndcg_10']:.3f}")
        print(f"  Diff: {row['diff_ndcg_10']:+.3f}")
        print(f"  History turns: {row['num_history_turns']} → Selected: {row['selected_turns']}")
        orig = row['original_query'][:100] if row['original_query'] else 'N/A'
        rewr = row['rewritten_query'][:150] if row['rewritten_query'] else 'N/A'
        print(f"  Original query: {orig}...")
        print(f"  Rewritten query: {rewr}...")


def main():
    print("="*80)
    print("PER-QUERY ANALYSIS: Targeted vs Standard Rewrite")
    print("="*80)
    
    all_results = {}
    
    for domain in DOMAINS:
        results = analyze_domain(domain)
        if results:
            all_results[domain] = results
    
    # Aggregate across domains
    print("\n" + "="*80)
    print("AGGREGATE ANALYSIS (ALL DOMAINS)")
    print("="*80)
    
    all_dfs = [r['df'] for r in all_results.values()]
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal queries: {len(combined)}")
        print(f"Mean diff nDCG@10: {combined['diff_ndcg_10'].mean():.4f}")
        print(f"Median diff nDCG@10: {combined['diff_ndcg_10'].median():.4f}")
        
        helps = combined[combined['diff_ndcg_10'] > 0.05]
        hurts = combined[combined['diff_ndcg_10'] < -0.05]
        print(f"\nTargeted HELPS: {len(helps)} ({100*len(helps)/len(combined):.1f}%)")
        print(f"Targeted HURTS: {len(hurts)} ({100*len(hurts)/len(combined):.1f}%)")
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE QUERIES")
    print("="*80)
    
    for domain in DOMAINS:
        if domain not in all_results:
            continue
        
        results = all_results[domain]
        print(f"\n{'='*60}")
        print(f"{domain.upper()}")
        print('='*60)
        
        print("\n--- Targeted HELPS (biggest gains) ---")
        show_examples(results, 'helps', n=3)
        
        print("\n--- Targeted HURTS (biggest losses) ---")
        show_examples(results, 'hurts', n=3)
        
        print("\n--- Both MISS (long conversations, room for improvement) ---")
        show_examples(results, 'both_miss', n=3)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
