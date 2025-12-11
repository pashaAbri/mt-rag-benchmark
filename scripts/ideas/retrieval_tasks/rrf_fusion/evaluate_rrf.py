#!/usr/bin/env python3
"""
Evaluate Reciprocal Rank Fusion (RRF) for Strategy Fusion.

Hypothesis:
RRF can robustly combine diverse retrieval strategies (lastturn, rewrite, questions)
without the computational cost of a Cross-Encoder (MonoT5).

Method:
1. Load retrieval results for all strategies.
2. Calculate RRF score for each document: Score(d) = sum(1 / (k + rank(d)))
3. Sort and take top 100.
4. Evaluate.
"""

import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

# Paths
RESULTS_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts' / 'elser' / 'results'
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'
INTERMEDIATE_DIR = script_dir / "intermediate"

# Config
DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
STRATEGIES = ['lastturn', 'rewrite', 'questions']
RRF_K = 60  # Standard constant for RRF

# Collection names mapping (needed for evaluation script format)
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'govt': 'mt-rag-govt-elser-512-100-20240611',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
}

def load_retrieval_results(domain: str, strategy: str):
    """Load retrieval results."""
    filename = f"elser_{domain}_{strategy}.jsonl"
    filepath = RESULTS_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: File not found {filepath}")
        return {}
    
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                contexts = data.get('contexts', [])
                if task_id:
                    results[task_id] = contexts
            except json.JSONDecodeError:
                continue
    return results

def compute_rrf(results_by_strategy):
    """
    Compute RRF scores for combined results.
    Returns: Dict[task_id, List[Dict]] (sorted documents)
    """
    fused_results = {}
    
    # Get all task_ids (intersection or union? Union maximizes recall)
    all_task_ids = set()
    for strat in results_by_strategy:
        all_task_ids.update(results_by_strategy[strat].keys())
        
    for task_id in all_task_ids:
        doc_scores = defaultdict(float)
        doc_info = {} # Keep track of doc metadata
        
        for strategy in STRATEGIES:
            if strategy not in results_by_strategy or task_id not in results_by_strategy[strategy]:
                continue
                
            ranked_docs = results_by_strategy[strategy][task_id]
            
            for rank, doc in enumerate(ranked_docs):
                doc_id = doc.get('document_id')
                if not doc_id: continue
                
                # RRF Formula: 1 / (k + rank)
                # rank is 0-indexed here, so rank+1 for 1-based rank
                score = 1.0 / (RRF_K + (rank + 1))
                doc_scores[doc_id] += score
                
                if doc_id not in doc_info:
                    doc_info[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format for output
        formatted_docs = []
        for doc_id, score in sorted_docs:
            original_doc = doc_info[doc_id]
            # Create a clean doc entry with RRF score
            new_doc = {
                'document_id': doc_id,
                'score': score, # RRF score
                'text': original_doc.get('text', ''),
                'title': original_doc.get('title', ''),
                'source': original_doc.get('source', '')
            }
            formatted_docs.append(new_doc)
            
        fused_results[task_id] = formatted_docs
        
    return fused_results

def save_and_evaluate(results, domain):
    """Save results and run evaluation."""
    output_file = INTERMEDIATE_DIR / f"rrf_{domain}.jsonl"
    evaluated_file = INTERMEDIATE_DIR / f"rrf_{domain}_evaluated.jsonl"
    
    collection_name = COLLECTION_NAMES.get(domain, domain)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for task_id, contexts in results.items():
            entry = {
                "task_id": task_id,
                "Collection": collection_name,
                "contexts": contexts[:100] # Top 100
            }
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved RRF results to {output_file}")
    
    # Evaluate
    print(f"Running evaluation for {domain}...")
    cmd = [
        "python", str(EVAL_SCRIPT_PATH),
        "--input_file", str(output_file),
        "--output_file", str(evaluated_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Evaluation complete: {evaluated_file}")
        
        # Print aggregate
        agg_file = evaluated_file.with_suffix('') 
        agg_file = Path(f"{agg_file}_aggregate.csv")
        if agg_file.exists():
            print("\nAggregate Results:")
            with open(agg_file, 'r') as f:
                print(f.read())
                
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def main():
    print("="*80)
    print("RRF Strategy Fusion Evaluation")
    print("="*80)
    
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    
    for domain in DOMAINS:
        print(f"\nProcessing {domain.upper()}...")
        
        results_by_strategy = {}
        for strategy in STRATEGIES:
            print(f"  Loading {strategy}...")
            results_by_strategy[strategy] = load_retrieval_results(domain, strategy)
            
        print("  Computing RRF...")
        fused_results = compute_rrf(results_by_strategy)
        print(f"  Fused {len(fused_results)} tasks")
        
        save_and_evaluate(fused_results, domain)

if __name__ == "__main__":
    main()
