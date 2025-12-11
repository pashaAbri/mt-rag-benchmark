#!/usr/bin/env python3
import json
import math
import csv
from pathlib import Path
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

DOMAINS = ['clapnq', 'govt']
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
        next(reader, None) # header
        for row in reader:
            if len(row) >= 2:
                qid, docid = row[0], row[1]
                score = int(row[2]) if len(row) > 2 else 1
                qrels[qid][docid] = score
    return qrels

def evaluate(run_file, qrels, k=10):
    recall_sum = 0
    ndcg_sum = 0
    count = 0
    
    with open(run_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                qid = data.get('task_id')
                if not qid: continue
                
                relevant_docs = qrels.get(qid)
                if not relevant_docs: continue
                
                retrieved = data.get('contexts', [])[:k]
                
                # Recall@k
                num_rel_retrieved = sum(1 for d in retrieved if d['document_id'] in relevant_docs)
                num_rel_total = len(relevant_docs)
                recall = num_rel_retrieved / num_rel_total if num_rel_total > 0 else 0
                
                # nDCG@k
                dcg = 0
                idcg = 0
                for i, doc in enumerate(retrieved):
                    doc_id = doc['document_id']
                    rel_score = relevant_docs.get(doc_id, 0)
                    if rel_score > 0:
                        dcg += rel_score / math.log2(i + 2)
                
                ideal_rels = sorted(relevant_docs.values(), reverse=True)
                for i, rel_score in enumerate(ideal_rels[:k]):
                    idcg += rel_score / math.log2(i + 2)
                    
                ndcg = dcg / idcg if idcg > 0 else 0
                
                recall_sum += recall
                ndcg_sum += ndcg
                count += 1
            except json.JSONDecodeError:
                continue
            
    return {
        'R@10': recall_sum / count if count > 0 else 0,
        'nDCG@10': ndcg_sum / count if count > 0 else 0
    }

def main():
    print(f"{'Domain':<8} | {'System':<8} | {'R@10':<8} | {'nDCG@10':<8}")
    print("-" * 45)
    
    for domain in DOMAINS:
        qrels = load_qrels(domain)
        for system in SYSTEMS:
            filename = f"{system}_{domain}_{STRATEGY}.jsonl"
            filepath = BASELINES_DIR / system / 'results' / filename
            
            if filepath.exists():
                metrics = evaluate(filepath, qrels)
                print(f"{domain:<8} | {system:<8} | {metrics['R@10']:.4f}   | {metrics['nDCG@10']:.4f}")
            else:
                print(f"{domain:<8} | {system:<8} | N/A      | N/A")

if __name__ == "__main__":
    main()
