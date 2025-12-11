#!/usr/bin/env python3
"""
Calculate the Theoretical Upper Bound (Oracle) for System Fusion across all domains.
"""
print("NEW VERSION RUNNING")

import json
import csv
from pathlib import Path
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
SYSTEMS = ['bm25', 'bge', 'elser']
STRATEGY = 'rewrite'
BASELINES_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts'

def load_qrels(domain):
    qrels_path = project_root / 'human' / 'retrieval_tasks' / domain / 'qrels' / 'dev.tsv'
    qrels = defaultdict(dict)
    
    if not qrels_path.exists():
        found = list(project_root.glob(f"**/{domain}/qrels/*.tsv"))
        if found: qrels_path = found[0]
        else: return {}

    with open(qrels_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                qid, docid = row[0], row[1]
                score = int(row[2]) if len(row) > 2 else 1
                qrels[qid][docid] = score
    return qrels

def load_results(domain, system):
    filename = f"{system}_{domain}_{STRATEGY}.jsonl"
    filepath = BASELINES_DIR / system / 'results' / filename
    
    results = {}
    if not filepath.exists():
        # print(f"Warning: {filepath} not found")
        return {}
        
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                qid = data.get('task_id')
                if qid:
                    results[qid] = [ctx.get('document_id') for ctx in data.get('contexts', [])]
            except json.JSONDecodeError:
                continue
    return results

def main():
    print(f"{'Domain':<10} | {'Tasks':<5} | {'Pool Oracle R@10':<18}")
    print("-" * 40)
    
    for domain in DOMAINS:
        qrels = load_qrels(domain)
        if not qrels:
            print(f"{domain:<10} | 0     | N/A")
            continue
            
        # Load all systems for this domain
        system_results = {}
        for system in SYSTEMS:
            system_results[system] = load_results(domain, system)
            
        # Calculate Pool Oracle
        task_ids = set(qrels.keys())
        pool_recall_sum = 0
        count = 0
        
        for qid in task_ids:
            # 1. Create Pool
            pool = set()
            for system in SYSTEMS:
                if qid in system_results[system]:
                    pool.update(system_results[system][qid])
            
            if not pool: continue
            
            # 2. Identify Relevant Docs in Pool
            relevant_docs = qrels[qid]
            relevant_in_pool = {doc_id for doc_id in pool if doc_id in relevant_docs}
            
            # 3. Calculate Oracle Recall@10
            best_retrieved_count = min(10, len(relevant_in_pool))
            total_relevant = len(relevant_docs)
            
            recall = best_retrieved_count / total_relevant if total_relevant > 0 else 0
            pool_recall_sum += recall
            count += 1
            
        avg_pool_recall = pool_recall_sum / count if count > 0 else 0
        print(f"{domain.upper():<10} | {count:<5} | {avg_pool_recall:.4f}")

if __name__ == "__main__":
    main()
