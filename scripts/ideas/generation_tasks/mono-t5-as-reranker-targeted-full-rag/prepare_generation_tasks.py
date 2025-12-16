"""
Prepare Generation Tasks from Mono-T5 Reranker-Targeted Retrieval Results

This script:
1. Loads the existing RAG.jsonl as a template (contains conversation history, targets, enrichments)
2. Loads the mono-t5 reranker-targeted retrieval results (3-strategy: lastturn + questions + targeted_rewrite)
3. Loads corpus files to get document text
4. Replaces contexts with top-K retrieved documents (with full text)
5. Outputs a new generation task file
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CORPUS_DIR = PROJECT_ROOT / "corpora" / "passage_level"
RETRIEVAL_RESULTS_DIR = PROJECT_ROOT / "scripts" / "ideas" / "retrieval_tasks" / "mono-t5-as-reranker-targeted" / "intermediate" / "using_targeted_rewrite_query"
RAG_TASKS_FILE = PROJECT_ROOT / "human" / "generation_tasks" / "RAG.jsonl"

# Domain to corpus file mapping
DOMAIN_CORPUS_FILES = {
    'clapnq': 'clapnq.jsonl',
    'cloud': 'cloud.jsonl',
    'fiqa': 'fiqa.jsonl',
    'govt': 'govt.jsonl'
}

# Domain to retrieval results file mapping (3-strategy: lastturn + questions + targeted_rewrite)
DOMAIN_RETRIEVAL_FILES = {
    'clapnq': 'reranked_lastturn_questions_targeted_rewrite_clapnq.jsonl',
    'cloud': 'reranked_lastturn_questions_targeted_rewrite_cloud.jsonl',
    'fiqa': 'reranked_lastturn_questions_targeted_rewrite_fiqa.jsonl',
    'govt': 'reranked_lastturn_questions_targeted_rewrite_govt.jsonl'
}

# Collection name to domain mapping
COLLECTION_TO_DOMAIN = {
    'mt-rag-clapnq-elser-512-100-20240503': 'clapnq',
    'mt-rag-ibmcloud-elser-512-100-20240502': 'cloud',
    'mt-rag-fiqa-beir-elser-512-100-20240501': 'fiqa',
    'mt-rag-govt-elser-512-100-20240611': 'govt'
}


def load_corpus(domain: str) -> Dict[str, Dict]:
    """Load corpus for a domain, returning doc_id -> document mapping."""
    corpus_file = CORPUS_DIR / DOMAIN_CORPUS_FILES[domain]
    
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    
    return corpus


def load_retrieval_results(domain: str) -> Dict[str, List[Dict]]:
    """Load retrieval results for a domain, returning task_id -> contexts mapping."""
    retrieval_file = RETRIEVAL_RESULTS_DIR / DOMAIN_RETRIEVAL_FILES[domain]
    
    if not retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval results not found: {retrieval_file}")
    
    results = {}
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            task_id = data['task_id']
            results[task_id] = data.get('contexts', [])
    
    return results


def load_rag_tasks() -> List[Dict[str, Any]]:
    """Load the original RAG.jsonl tasks."""
    tasks = []
    with open(RAG_TASKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def get_domain_from_collection(collection: str) -> str:
    """Map collection name to domain."""
    return COLLECTION_TO_DOMAIN.get(collection, None)


def enrich_contexts_with_text(
    contexts: List[Dict], 
    corpus: Dict[str, Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Enrich retrieval contexts with document text from corpus.
    
    Args:
        contexts: List of {"document_id": ..., "score": ...}
        corpus: Document ID to document mapping
        top_k: Number of top documents to include
        
    Returns:
        List of contexts with full document text
    """
    enriched = []
    
    for ctx in contexts[:top_k]:
        doc_id = ctx['document_id']
        score = ctx['score']
        
        if doc_id in corpus:
            doc = corpus[doc_id]
            enriched.append({
                'document_id': doc_id,
                'score': score,
                'text': doc.get('text', ''),
                'title': doc.get('title', ''),
                'source': doc.get('url', '')
            })
        else:
            print(f"Warning: Document {doc_id} not found in corpus")
            enriched.append({
                'document_id': doc_id,
                'score': score,
                'text': '',
                'title': '',
                'source': ''
            })
    
    return enriched


def main():
    parser = argparse.ArgumentParser(
        description='Prepare generation tasks from mono-t5 reranker-targeted retrieval results'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(PROJECT_ROOT / "scripts" / "ideas" / "generation_tasks" / "mono-t5-as-reranker-targeted-full-rag" / "mono_t5_targeted_RAG.jsonl"),
        help='Output file path'
    )
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=5,
        help='Number of top documents to include (default: 5)'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preparing Generation Tasks from Mono-T5 Reranker-Targeted")
    print("=" * 60)
    
    # Load corpora for all domains
    print("\n1. Loading corpus files...")
    corpora = {}
    for domain in DOMAIN_CORPUS_FILES.keys():
        print(f"   Loading {domain} corpus...")
        corpora[domain] = load_corpus(domain)
        print(f"   Loaded {len(corpora[domain])} documents")
    
    # Load retrieval results for all domains
    print("\n2. Loading retrieval results...")
    retrieval_results = {}
    for domain in DOMAIN_RETRIEVAL_FILES.keys():
        print(f"   Loading {domain} retrieval results...")
        retrieval_results[domain] = load_retrieval_results(domain)
        print(f"   Loaded {len(retrieval_results[domain])} tasks")
    
    # Combine all retrieval results by task_id
    all_retrieval = {}
    for domain, results in retrieval_results.items():
        for task_id, contexts in results.items():
            all_retrieval[task_id] = {'domain': domain, 'contexts': contexts}
    
    print(f"\n   Total retrieval results: {len(all_retrieval)} tasks")
    
    # Load RAG tasks template
    print("\n3. Loading RAG.jsonl template...")
    rag_tasks = load_rag_tasks()
    print(f"   Loaded {len(rag_tasks)} tasks")
    
    # Process tasks
    print(f"\n4. Processing tasks (top-{args.top_k} documents)...")
    output_tasks = []
    missing_retrieval = 0
    
    for task in tqdm(rag_tasks, desc="Processing"):
        task_id = task['task_id']
        collection = task.get('Collection', '')
        
        # Get domain from collection name
        domain = get_domain_from_collection(collection)
        
        if task_id in all_retrieval:
            # Get retrieval results
            retrieval_info = all_retrieval[task_id]
            retrieval_domain = retrieval_info['domain']
            contexts = retrieval_info['contexts']
            
            # Enrich with document text
            enriched_contexts = enrich_contexts_with_text(
                contexts, 
                corpora[retrieval_domain],
                top_k=args.top_k
            )
            
            # Create new task with updated contexts
            new_task = task.copy()
            new_task['contexts'] = enriched_contexts
            new_task['retrieval_method'] = 'mono-t5-reranker-targeted-3strategy'
            output_tasks.append(new_task)
        else:
            missing_retrieval += 1
            # Keep original task but mark it
            new_task = task.copy()
            new_task['retrieval_method'] = 'original-elser'
            output_tasks.append(new_task)
    
    if missing_retrieval > 0:
        print(f"\n   Warning: {missing_retrieval} tasks missing from retrieval results (kept original contexts)")
    
    # Save output
    print(f"\n5. Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for task in output_tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"\n✓ Complete!")
    print(f"  Output: {args.output}")
    print(f"  Tasks: {len(output_tasks)}")
    print(f"  Documents per task: top-{args.top_k}")
    
    # Summary statistics
    retrieval_methods = {}
    for task in output_tasks:
        method = task.get('retrieval_method', 'unknown')
        retrieval_methods[method] = retrieval_methods.get(method, 0) + 1
    
    print(f"\n  Retrieval method breakdown:")
    for method, count in retrieval_methods.items():
        print(f"    {method}: {count}")


if __name__ == "__main__":
    main()
