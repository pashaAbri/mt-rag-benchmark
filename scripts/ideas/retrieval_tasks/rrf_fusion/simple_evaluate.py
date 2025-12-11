#!/usr/bin/env python3
import json
import math
import csv
from pathlib import Path
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

def load_qrels(domain):
    qrels_path = project_root / 'human' / 'retrieval_tasks' / domain / 'qrels' / 'dev.tsv'
    qrels = defaultdict(dict)
    
    if not qrels_path.exists():
        # Try finding it
        found = list(project_root.glob(f"**/{domain}/qrels/*.tsv"))
        if found:
            qrels_path = found[0]
        else:
            print(f"Qrels not found for {domain}")
            return {}

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
            data = json.loads(line)
            qid = data['task_id']
            
            # Extract task_id from query_id if needed, but qrels usually match query_id
            # However, in this dataset, qrels might be keyed by query_id (with turn)
            # Let's check if qid is in qrels
            if qid not in qrels:
                # Try stripping turn suffix if qrels are task-level (unlikely for retrieval)
                # But wait, retrieval qrels usually are query-level.
                continue
                
            relevant_docs = qrels[qid]
            if not relevant_docs: continue
            
            retrieved = data['contexts'][:k]
            
            # Recall@k
            num_rel_retrieved = sum(1 for d in retrieved if d['document_id'] in relevant_docs)
            num_rel_total = len(relevant_docs)
            recall = num_rel_retrieved / num_rel_total if num_rel_total > 0 else 0
            
            # nDCG@k
            dcg = 0
            idcg = 0
            
            # DCG
            for i, doc in enumerate(retrieved):
                doc_id = doc['document_id']
                rel_score = relevant_docs.get(doc_id, 0)
                if rel_score > 0:
                    dcg += rel_score / math.log2(i + 2)
            
            # IDCG
            ideal_rels = sorted(relevant_docs.values(), reverse=True)
            for i, rel_score in enumerate(ideal_rels[:k]):
                idcg += rel_score / math.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
            
            recall_sum += recall
            ndcg_sum += ndcg
            count += 1
            
    return {
        'R@10': recall_sum / count if count > 0 else 0,
        'nDCG@10': ndcg_sum / count if count > 0 else 0,
        'count': count
    }

def main():
    print(f"{'Domain':<10} | {'R@10':<8} | {'nDCG@10':<8} | {'Count':<5}")
    print("-" * 40)
    
    for domain in DOMAINS:
        qrels = load_qrels(domain)
        run_file = script_dir / "intermediate" / f"weighted_rrf_systems_{domain}.jsonl"
        
        if not run_file.exists():
            print(f"{domain:<10} | N/A      | N/A      | 0")
            continue
            
        metrics = evaluate(run_file, qrels)
        print(f"{domain.upper():<10} | {metrics['R@10']:.4f}   | {metrics['nDCG@10']:.4f}   | {metrics['count']}")

if __name__ == "__main__":
    main()
