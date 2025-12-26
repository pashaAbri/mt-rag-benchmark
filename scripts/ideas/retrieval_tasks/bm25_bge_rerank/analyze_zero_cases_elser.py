#!/usr/bin/env python3
"""
Analyze whether the two-stage BM25+ELSER approach helps the 98 zero-score cases.
"""

import json
import csv
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

# Load zero-score cases
zero_cases_file = project_root / "scripts/ideas/retrieval_tasks/oracle-v2/new-strategies/new-strategies/zero_score_cases.csv"

zero_task_ids = set()
zero_cases_by_domain = {}

with open(zero_cases_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        task_id = row['task_id']
        domain = row['domain']
        zero_task_ids.add(task_id)
        if domain not in zero_cases_by_domain:
            zero_cases_by_domain[domain] = []
        zero_cases_by_domain[domain].append({
            'task_id': task_id,
            'query': row['query_text'],
            'turn': row['turn_id'],
            'type': row['question_type']
        })

print(f"Total zero-score cases: {len(zero_task_ids)}")
print(f"By domain: {', '.join(f'{d}: {len(cases)}' for d, cases in zero_cases_by_domain.items())}")
print()

# Check BM25+ELSER results (rewrite only since that's what we ran with @500)
results_dir = script_dir / "results"
domains = ['clapnq', 'cloud', 'fiqa', 'govt']
query_type = 'rewrite'  # We only ran rewrite

helped_cases = []
still_zero = []

for domain in domains:
    domain_zero_ids = {c['task_id'] for c in zero_cases_by_domain.get(domain, [])}
    
    # Load evaluated results for BM25+ELSER
    eval_file = results_dir / f"bm25_elser_rerank_{domain}_{query_type}_evaluated.jsonl"
    
    if not eval_file.exists():
        print(f"Warning: {eval_file.name} not found")
        continue
    
    with open(eval_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data['task_id']
            
            if task_id in domain_zero_ids:
                # Check if we got any recall
                scores = data.get('retriever_scores', {})
                recall_10 = scores.get('recall_10', 0)
                ndcg_10 = scores.get('ndcg_cut_10', 0)
                
                # Get query text from zero cases
                query_info = next((c for c in zero_cases_by_domain[domain] if c['task_id'] == task_id), {})
                
                case_info = {
                    'task_id': task_id,
                    'domain': domain,
                    'recall_10': recall_10,
                    'ndcg_10': ndcg_10,
                    'query': query_info.get('query', '')[:50],
                    'type': query_info.get('type', '')
                }
                
                if recall_10 > 0:
                    helped_cases.append(case_info)
                else:
                    still_zero.append(case_info)

# Aggregate results
print("=" * 70)
print("ZERO-CASE RECOVERY ANALYSIS: BM25@500 + ELSER (rewrite query)")
print("=" * 70)

# Count unique task_ids that were helped
helped_task_ids = set(c['task_id'] for c in helped_cases)
print(f"\n✅ Zero cases RECOVERED: {len(helped_task_ids)} / {len(zero_task_ids)} ({100*len(helped_task_ids)/len(zero_task_ids):.1f}%)")

if helped_cases:
    print("\nRecovered cases (sorted by recall):")
    print("-" * 70)
    for case in sorted(helped_cases, key=lambda x: -x['recall_10'])[:20]:
        print(f"  [{case['domain']:6}] R@10={case['recall_10']:.3f} nDCG@10={case['ndcg_10']:.3f} | {case['type']:12} | {case['query'][:40]}...")

# Show how many remain zero
remaining_zero = zero_task_ids - helped_task_ids
print(f"\n❌ Zero cases STILL ZERO: {len(remaining_zero)} / {len(zero_task_ids)} ({100*len(remaining_zero)/len(zero_task_ids):.1f}%)")

# Summary by domain
print("\n" + "-" * 70)
print("Recovery by domain:")
for domain in domains:
    domain_helped = [c for c in helped_cases if c['domain'] == domain]
    domain_total = len(zero_cases_by_domain.get(domain, []))
    if domain_total > 0:
        print(f"  {domain:8}: {len(domain_helped):2} / {domain_total:2} recovered ({100*len(domain_helped)/domain_total:.1f}%)")

# Compare with BM25+BGE
print("\n" + "=" * 70)
print("COMPARISON: BM25+ELSER vs BM25+BGE (zero-case recovery)")
print("=" * 70)

# Check BM25+BGE results for the same query type
bge_helped = set()
for domain in domains:
    domain_zero_ids = {c['task_id'] for c in zero_cases_by_domain.get(domain, [])}
    
    eval_file = results_dir / f"bm25_bge_rerank_{domain}_{query_type}_evaluated.jsonl"
    
    if not eval_file.exists():
        continue
    
    with open(eval_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data['task_id']
            
            if task_id in domain_zero_ids:
                scores = data.get('retriever_scores', {})
                recall_10 = scores.get('recall_10', 0)
                if recall_10 > 0:
                    bge_helped.add(task_id)

print(f"\nBM25+BGE recovered:   {len(bge_helped)} / {len(zero_task_ids)} ({100*len(bge_helped)/len(zero_task_ids):.1f}%)")
print(f"BM25+ELSER recovered: {len(helped_task_ids)} / {len(zero_task_ids)} ({100*len(helped_task_ids)/len(zero_task_ids):.1f}%)")

# Cases recovered by ELSER but not BGE
elser_only = helped_task_ids - bge_helped
bge_only = bge_helped - helped_task_ids
both = helped_task_ids & bge_helped

print(f"\nRecovered by both:      {len(both)}")
print(f"Recovered by ELSER only: {len(elser_only)}")
print(f"Recovered by BGE only:   {len(bge_only)}")

print("\n" + "=" * 70)

