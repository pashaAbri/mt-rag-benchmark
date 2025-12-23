#!/usr/bin/env python3
"""
Evaluate L+R+S (lastturn + rewrite + summary) fusion results with per-query scores.

This script evaluates the fusion_reranked ELSER results and saves them with
retriever_scores for each query, enabling comparison with L+Q+R fusion.

Usage:
    python evaluate_lrs_fusion.py
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pytrec_eval
except ImportError:
    print("Installing pytrec_eval...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pytrec_eval'])
    import pytrec_eval

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pandas'])
    import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FUSION_RESULTS_DIR = SCRIPT_DIR / "fusion_rerank_results"
OUTPUT_DIR = SCRIPT_DIR / "lrs_evaluated"
QRELS_DIR = PROJECT_ROOT / "human" / "retrieval_tasks"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']


def load_qrels(domain: str) -> Dict[str, Dict[str, int]]:
    """Load qrels for a domain."""
    qrels_file = QRELS_DIR / domain / "qrels" / "dev.tsv"
    
    if not qrels_file.exists():
        print(f"Warning: Qrels file not found: {qrels_file}")
        return {}
    
    qrels = {}
    with open(qrels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        next(reader)  # Skip header
        
        for row in reader:
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][corpus_id] = score
    
    return qrels


def load_fusion_results(domain: str) -> List[dict]:
    """Load fusion results for a domain."""
    # Try ELSER version first
    fusion_file = FUSION_RESULTS_DIR / f"fusion_reranked_{domain}_elser.jsonl"
    
    if not fusion_file.exists():
        # Fall back to default
        fusion_file = FUSION_RESULTS_DIR / f"fusion_reranked_{domain}.jsonl"
    
    if not fusion_file.exists():
        print(f"Warning: Fusion results not found for {domain}")
        return []
    
    results = []
    with open(fusion_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    return results


def prepare_results_dict(results: List[dict]) -> Dict[str, Dict[str, float]]:
    """Convert results list to dict format for pytrec_eval."""
    results_dict = {}
    
    for item in results:
        query_id = item["task_id"]
        doc_scores = {}
        
        for ctx in item.get("contexts", []):
            doc_id = ctx["document_id"]
            score = ctx["score"]
            doc_scores[doc_id] = score
        
        results_dict[query_id] = doc_scores
    
    return results_dict


def evaluate(
    qrels: Dict[str, Dict[str, int]], 
    results: Dict[str, Dict[str, float]], 
    k_values: List[int] = [1, 3, 5, 10]
) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, float]]:
    """Evaluate retrieval results using pytrec_eval."""
    
    if len(results) == 0:
        return {}, {}, {}
    
    # Filter results to only include queries with qrels
    filtered_results = {qid: docs for qid, docs in results.items() if qid in qrels}
    
    if len(filtered_results) == 0:
        return {}, {}, {}
    
    # Set up evaluation metrics
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, recall_string})
    scores = evaluator.evaluate(filtered_results)
    
    # Aggregate scores
    ndcg_agg = {f"nDCG@{k}": 0.0 for k in k_values}
    recall_agg = {f"Recall@{k}": 0.0 for k in k_values}
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg_agg[f"nDCG@{k}"] += scores[query_id][f"ndcg_cut_{k}"]
            recall_agg[f"Recall@{k}"] += scores[query_id][f"recall_{k}"]
    
    n_queries = len(scores)
    for k in k_values:
        ndcg_agg[f"nDCG@{k}"] = round(ndcg_agg[f"nDCG@{k}"] / n_queries, 5)
        recall_agg[f"Recall@{k}"] = round(recall_agg[f"Recall@{k}"] / n_queries, 5)
    
    return scores, ndcg_agg, recall_agg


def enrich_results(results: List[dict], scores_per_query: Dict[str, dict]) -> List[dict]:
    """Add retriever_scores to each result."""
    enriched = []
    
    for item in results:
        query_id = item["task_id"]
        item_copy = item.copy()
        
        if query_id in scores_per_query:
            item_copy["retriever_scores"] = scores_per_query[query_id]
        else:
            item_copy["retriever_scores"] = {}
        
        enriched.append(item_copy)
    
    return enriched


def main():
    print("=" * 70)
    print("Evaluating L+R+S Fusion Results")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_scores = {}
    domain_aggregates = []
    
    for domain in DOMAINS:
        print(f"\nProcessing {domain}...")
        
        # Load data
        qrels = load_qrels(domain)
        fusion_results = load_fusion_results(domain)
        
        if not qrels or not fusion_results:
            print(f"  Skipping {domain} - missing data")
            continue
        
        print(f"  Loaded {len(qrels)} qrels, {len(fusion_results)} fusion results")
        
        # Prepare and evaluate
        results_dict = prepare_results_dict(fusion_results)
        scores_per_query, ndcg_agg, recall_agg = evaluate(qrels, results_dict)
        
        print(f"  Evaluated {len(scores_per_query)} queries")
        print(f"  R@5: {recall_agg.get('Recall@5', 'N/A'):.4f}, R@10: {recall_agg.get('Recall@10', 'N/A'):.4f}")
        
        # Enrich results
        enriched = enrich_results(fusion_results, scores_per_query)
        
        # Save per-domain results
        output_file = OUTPUT_DIR / f"fusion_lrs_{domain}_elser_evaluated.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in enriched:
                f.write(json.dumps(item) + '\n')
        
        print(f"  Saved to {output_file}")
        
        # Collect for combined file
        all_results.extend(enriched)
        all_scores.update(scores_per_query)
        
        domain_aggregates.append({
            'domain': domain,
            'count': len(scores_per_query),
            **ndcg_agg,
            **recall_agg,
        })
    
    # Save combined results
    print(f"\n{'='*70}")
    print("Saving combined results...")
    
    combined_file = OUTPUT_DIR / "fusion_lrs_all_elser_evaluated.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item) + '\n')
    print(f"  Combined results: {combined_file}")
    
    # Calculate overall aggregates
    total_count = sum(d['count'] for d in domain_aggregates)
    
    weighted_metrics = {}
    for metric in ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 
                   'nDCG@1', 'nDCG@3', 'nDCG@5', 'nDCG@10']:
        weighted_sum = sum(d[metric] * d['count'] for d in domain_aggregates)
        weighted_metrics[metric] = weighted_sum / total_count
    
    domain_aggregates.append({
        'domain': 'ALL',
        'count': total_count,
        **weighted_metrics,
    })
    
    # Save aggregate CSV
    aggregate_file = OUTPUT_DIR / "fusion_lrs_aggregate.csv"
    df = pd.DataFrame(domain_aggregates)
    df.to_csv(aggregate_file, index=False)
    print(f"  Aggregate metrics: {aggregate_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("L+R+S Fusion Evaluation Summary")
    print(f"{'='*70}")
    print(f"\nTotal queries evaluated: {total_count}")
    print(f"\nAggregate Metrics:")
    print(f"  R@5:  {weighted_metrics['Recall@5']:.4f}")
    print(f"  R@10: {weighted_metrics['Recall@10']:.4f}")
    print(f"  nDCG@5:  {weighted_metrics['nDCG@5']:.4f}")
    print(f"  nDCG@10: {weighted_metrics['nDCG@10']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

