import os
import json
import math
from typing import List, Dict, Any

def load_run_file(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data.get('task_id')
            contexts = data.get('contexts', [])[:10] # Top 10
            results[task_id] = contexts
    return results

def load_queries(domain: str) -> Dict[str, str]:
    queries = {}
    task_dir = f"../../../../cleaned_data/tasks/{domain}"
    if not os.path.exists(task_dir):
        print(f"Task directory not found: {task_dir}")
        return {}
        
    for filename in os.listdir(task_dir):
        if filename.endswith(".json"):
            with open(os.path.join(task_dir, filename), 'r') as f:
                data = json.load(f)
                task_id = data.get('task_id')
                # Query text is stored in user.text, not question
                user_data = data.get('user', {})
                query_text = user_data.get('text', '') if isinstance(user_data, dict) else ''
                if task_id and query_text:
                    queries[task_id] = query_text
    return queries

def compute_predicted_ndcg(relevance_labels: List[int]) -> float:
    k = len(relevance_labels)
    if k == 0: return 0.0
    
    dcg = 0.0
    for i, rel in enumerate(relevance_labels):
        if rel > 0:
            dcg += 1.0 / math.log2(i + 2)
            
    ideal_labels = sorted(relevance_labels, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_labels):
        if rel > 0:
            idcg += 1.0 / math.log2(i + 2)
            
    return dcg / idcg if idcg > 0 else 0.0

