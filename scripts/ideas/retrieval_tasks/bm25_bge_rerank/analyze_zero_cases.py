#!/usr/bin/env python3
"""
Analyze whether the two-stage BM25+BGE approach helps the 98 zero-score cases.
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

# Check our results
results_dir = script_dir / "results"
domains = ['clapnq', 'cloud', 'fiqa', 'govt']
query_types = ['lastturn', 'rewrite', 'questions']

helped_cases = []
still_zero = []

for domain in domains:
    domain_zero_ids = {c['task_id'] for c in zero_cases_by_domain.get(domain, [])}
    
    for query_type in query_types:
        # Load evaluated results
        eval_file = results_dir / f"bm25_bge_rerank_{domain}_{query_type}_evaluated.jsonl"
        
        if not eval_file.exists():
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
                    
                    case_info = {
                        'task_id': task_id,
                        'domain': domain,
                        'query_type': query_type,
                        'recall_10': recall_10,
                        'ndcg_10': ndcg_10
                    }
                    
                    if recall_10 > 0:
                        helped_cases.append(case_info)
                    else:
                        still_zero.append(case_info)

# Aggregate results
print("=" * 70)
print("ZERO-CASE RECOVERY ANALYSIS")
print("=" * 70)

# Count unique task_ids that were helped (across any query type)
helped_task_ids = set(c['task_id'] for c in helped_cases)
print(f"\n✅ Zero cases RECOVERED by BM25+BGE: {len(helped_task_ids)} / {len(zero_task_ids)} ({100*len(helped_task_ids)/len(zero_task_ids):.1f}%)")

if helped_cases:
    print("\nRecovered cases:")
    for case in sorted(helped_cases, key=lambda x: -x['recall_10']):
        print(f"  {case['task_id'][:30]}... [{case['domain']}/{case['query_type']}] R@10={case['recall_10']:.3f}, nDCG@10={case['ndcg_10']:.3f}")

# Show how many remain zero
remaining_zero = zero_task_ids - helped_task_ids
print(f"\n❌ Zero cases STILL ZERO: {len(remaining_zero)} / {len(zero_task_ids)} ({100*len(remaining_zero)/len(zero_task_ids):.1f}%)")

# Summary by query type
print("\n" + "-" * 70)
print("Recovery by query type:")
for qt in query_types:
    qt_helped = [c for c in helped_cases if c['query_type'] == qt]
    qt_task_ids = set(c['task_id'] for c in qt_helped)
    print(f"  {qt}: {len(qt_task_ids)} cases recovered")

print("\n" + "=" * 70)

