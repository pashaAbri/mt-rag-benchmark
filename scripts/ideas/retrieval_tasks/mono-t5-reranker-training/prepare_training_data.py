#!/usr/bin/env python3
"""
Prepare training data for MonoT5 relevance scorer.

This script:
1. Loads Qrels (positive examples)
2. Loads retrieval results (for hard negatives)
3. Loads queries and corpus documents
4. Splits data by task_id (not query_id) to avoid data leakage
5. Creates train/val/test splits
6. Saves data in MonoT5 format: "Query: {query} Document: {doc} Relevant:" -> "true"/"false"
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import argparse

# Constants
DOMAINS = ['clapnq', 'fiqa', 'govt', 'cloud']
STRATEGIES = ['rewrite', 'lastturn', 'questions']
RETRIEVAL_METHOD = 'elser'  # We'll use ELSER results for hard negatives

# MonoT5 input format
INPUT_FORMAT = "Query: {query} Document: {document} Relevant:"
OUTPUT_TRUE = "true"
OUTPUT_FALSE = "false"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Negative sampling ratio (positives : negatives)
NEGATIVE_RATIO = 4  # 1 positive : 4 negatives


def extract_task_id(query_id: str) -> str:
    """Extract task_id from query_id (remove turn number)."""
    if '<::>' in query_id:
        return query_id.split('<::>')[0]
    return query_id


def load_qrels(qrels_path: Path) -> Dict[str, Set[str]]:
    """
    Load Qrels file.
    
    Returns:
        Dict mapping query_id -> set of relevant corpus_ids
    """
    qrels = defaultdict(set)
    
    if not qrels_path.exists():
        print(f"ERROR: Qrels file not found: {qrels_path}")
        print(f"  Absolute path: {qrels_path.resolve()}")
        return qrels
    
    count = 0
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                query_id = parts[0]
                corpus_id = parts[1]
                qrels[query_id].add(corpus_id)
                count += 1
    
    print(f"    Loaded {count} qrel entries")
    return qrels


def load_queries(queries_path: Path) -> Dict[str, str]:
    """Load queries from JSONL file."""
    queries = {}
    
    if not queries_path.exists():
        print(f"ERROR: Queries file not found: {queries_path}")
        print(f"  Absolute path: {queries_path.resolve()}")
        return queries
    
    count = 0
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query_id = data.get('_id')
            query_text = data.get('text', '')
            if query_id and query_text:
                queries[query_id] = query_text
                count += 1
    
    print(f"    Loaded {count} queries")
    return queries


def load_corpus(corpus_path: Path) -> Dict[str, str]:
    """Load corpus documents from JSONL file."""
    corpus = {}
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus file not found: {corpus_path}")
        print(f"  Absolute path: {corpus_path.resolve()}")
        return corpus
    
    count = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('_id')
            # Try different possible fields for document text
            doc_text = data.get('text', '') or data.get('contents', '') or data.get('body', '')
            if doc_id and doc_text:
                corpus[doc_id] = doc_text
                count += 1
    
    print(f"    Loaded {count} corpus documents")
    return corpus


def load_retrieval_results(results_path: Path) -> Dict[str, List[Dict]]:
    """
    Load retrieval results.
    
    Returns:
        Dict mapping query_id -> list of retrieved documents with document_id and text
    """
    results = {}
    
    if not results_path.exists():
        print(f"ERROR: Retrieval results file not found: {results_path}")
        print(f"  Absolute path: {results_path.resolve()}")
        return results
    
    count = 0
    total_docs = 0
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query_id = data.get('task_id')  # Note: retrieval results use 'task_id' field
            contexts = data.get('contexts', [])
            
            if query_id:
                results[query_id] = []
                for ctx in contexts:
                    doc_id = ctx.get('document_id')
                    doc_text = ctx.get('text', '')
                    if doc_id and doc_text:
                        results[query_id].append({
                            'document_id': doc_id,
                            'text': doc_text
                        })
                        total_docs += 1
                if results[query_id]:
                    count += 1
    
    print(f"    Loaded {count} queries with {total_docs} total retrieved documents")
    return results


def create_examples(
    query_id: str,
    query_text: str,
    qrels: Dict[str, Set[str]],
    retrieval_results: Dict[str, List[Dict]],
    corpus: Dict[str, str],
    domain: str,
    strategy: str,
    negative_ratio: int = NEGATIVE_RATIO
) -> List[Dict]:
    """
    Create training examples for a single query.
    
    Returns:
        List of examples with references (query_id, document_id, label, task_id, domain, strategy)
    """
    examples = []
    task_id = extract_task_id(query_id)
    
    # Get relevant document IDs for this query
    relevant_doc_ids = qrels.get(query_id, set())
    
    # Get retrieved documents for this query
    retrieved_docs = retrieval_results.get(query_id, [])
    
    # Create positive examples
    for doc_id in relevant_doc_ids:
        # Verify document exists (either in corpus or retrieval results)
        doc_exists = False
        if doc_id in corpus:
            doc_exists = True
        else:
            # Check retrieval results
            for ret_doc in retrieved_docs:
                if ret_doc['document_id'] == doc_id:
                    doc_exists = True
                    break
        
        if doc_exists:
            examples.append({
                'query_id': query_id,
                'document_id': doc_id,
                'label': 1,
                'task_id': task_id,
                'domain': domain,
                'strategy': strategy
            })
    
    # Create negative examples (hard negatives from retrieval results)
    negative_count = len(relevant_doc_ids) * negative_ratio
    negative_added = 0
    
    for ret_doc in retrieved_docs:
        if negative_added >= negative_count:
            break
        
        doc_id = ret_doc['document_id']
        # Skip if this document is actually relevant
        if doc_id in relevant_doc_ids:
            continue
        
        # Document text exists (we already have it in ret_doc)
        examples.append({
            'query_id': query_id,
            'document_id': doc_id,
            'label': 0,
            'task_id': task_id,
            'domain': domain,
            'strategy': strategy
        })
        negative_added += 1
    
    return examples


def split_by_task(examples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split examples by task_id to avoid data leakage.
    
    Ensures all examples from the same conversation (task_id) stay in the same split.
    
    Returns:
        (train_examples, val_examples, test_examples)
    """
    # Group examples by task_id (use stored task_id from examples)
    examples_by_task = defaultdict(list)
    for ex in examples:
        task_id = ex.get('task_id')
        if not task_id:
            # Fallback: extract from query_id if not stored
            task_id = extract_task_id(ex['query_id'])
        examples_by_task[task_id].append(ex)
    
    # Get unique task IDs and shuffle
    task_ids = list(examples_by_task.keys())
    random.shuffle(task_ids)
    
    # Calculate split sizes
    n_tasks = len(task_ids)
    n_train = int(n_tasks * TRAIN_RATIO)
    n_val = int(n_tasks * VAL_RATIO)
    n_test = n_tasks - n_train - n_val
    
    # Split task IDs
    train_task_ids = set(task_ids[:n_train])
    val_task_ids = set(task_ids[n_train:n_train + n_val])
    test_task_ids = set(task_ids[n_train + n_val:])
    
    # Validate no overlap
    assert len(train_task_ids & val_task_ids) == 0, "Overlap between train and val task_ids!"
    assert len(train_task_ids & test_task_ids) == 0, "Overlap between train and test task_ids!"
    assert len(val_task_ids & test_task_ids) == 0, "Overlap between val and test task_ids!"
    assert len(train_task_ids) + len(val_task_ids) + len(test_task_ids) == n_tasks, "Task IDs lost in split!"
    
    # Split examples
    train_examples = []
    val_examples = []
    test_examples = []
    
    for task_id, task_examples in examples_by_task.items():
        if task_id in train_task_ids:
            train_examples.extend(task_examples)
        elif task_id in val_task_ids:
            val_examples.extend(task_examples)
        elif task_id in test_task_ids:
            test_examples.extend(task_examples)
        else:
            raise ValueError(f"Task {task_id} not assigned to any split!")
    
    # Validate all examples assigned
    assert len(train_examples) + len(val_examples) + len(test_examples) == len(examples), \
        f"Example count mismatch: {len(train_examples) + len(val_examples) + len(test_examples)} != {len(examples)}"
    
    # Validate no query_id appears in multiple splits
    train_query_ids = {ex['query_id'] for ex in train_examples}
    val_query_ids = {ex['query_id'] for ex in val_examples}
    test_query_ids = {ex['query_id'] for ex in test_examples}
    
    assert len(train_query_ids & val_query_ids) == 0, "Query IDs appear in both train and val!"
    assert len(train_query_ids & test_query_ids) == 0, "Query IDs appear in both train and test!"
    assert len(val_query_ids & test_query_ids) == 0, "Query IDs appear in both val and test!"
    
    # Print validation statistics
    print(f"  Split validation:")
    print(f"    Tasks: {n_tasks} total ({len(train_task_ids)} train, {len(val_task_ids)} val, {len(test_task_ids)} test)")
    print(f"    Unique queries: {len(train_query_ids)} train, {len(val_query_ids)} val, {len(test_query_ids)} test")
    print(f"    No overlap between splits: ✓")
    
    return train_examples, val_examples, test_examples


def verify_no_leakage(train_examples: List[Dict], val_examples: List[Dict], test_examples: List[Dict]) -> bool:
    """
    Verify that there's no data leakage between splits.
    
    Returns:
        True if no leakage detected, False otherwise
    """
    # Extract all task_ids and query_ids from each split
    train_task_ids = {ex['task_id'] for ex in train_examples}
    val_task_ids = {ex['task_id'] for ex in val_examples}
    test_task_ids = {ex['task_id'] for ex in test_examples}
    
    train_query_ids = {ex['query_id'] for ex in train_examples}
    val_query_ids = {ex['query_id'] for ex in val_examples}
    test_query_ids = {ex['query_id'] for ex in test_examples}
    
    # Check for overlaps
    task_overlaps = {
        'train_val': train_task_ids & val_task_ids,
        'train_test': train_task_ids & test_task_ids,
        'val_test': val_task_ids & test_task_ids
    }
    
    query_overlaps = {
        'train_val': train_query_ids & val_query_ids,
        'train_test': train_query_ids & test_query_ids,
        'val_test': val_query_ids & test_query_ids
    }
    
    # Check query-document pairs (most strict check)
    train_pairs = {(ex['query_id'], ex['document_id']) for ex in train_examples}
    val_pairs = {(ex['query_id'], ex['document_id']) for ex in val_examples}
    test_pairs = {(ex['query_id'], ex['document_id']) for ex in test_examples}
    
    pair_overlaps = {
        'train_val': train_pairs & val_pairs,
        'train_test': train_pairs & test_pairs,
        'val_test': val_pairs & test_pairs
    }
    
    # Report any issues
    has_leakage = False
    for split_pair, overlapping_tasks in task_overlaps.items():
        if overlapping_tasks:
            print(f"  ERROR: {len(overlapping_tasks)} task_ids overlap between {split_pair}")
            has_leakage = True
    
    for split_pair, overlapping_queries in query_overlaps.items():
        if overlapping_queries:
            print(f"  ERROR: {len(overlapping_queries)} query_ids overlap between {split_pair}")
            has_leakage = True
    
    for split_pair, overlapping_pairs in pair_overlaps.items():
        if overlapping_pairs:
            print(f"  ERROR: {len(overlapping_pairs)} query-document pairs overlap between {split_pair}")
            has_leakage = True
    
    if not has_leakage:
        print(f"  ✓ No data leakage detected")
        print(f"    Task IDs: {len(train_task_ids)} train, {len(val_task_ids)} val, {len(test_task_ids)} test (all disjoint)")
        print(f"    Query IDs: {len(train_query_ids)} train, {len(val_query_ids)} val, {len(test_query_ids)} test (all disjoint)")
        print(f"    Query-Document Pairs: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test (all disjoint)")
    
    return not has_leakage


def save_examples(examples: List[Dict], output_path: Path):
    """Save examples to JSONL file (as references only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            # Save only references - actual text will be loaded at training time
            f.write(json.dumps({
                'query_id': ex['query_id'],
                'document_id': ex['document_id'],
                'label': ex['label'],
                'task_id': ex['task_id'],
                'domain': ex['domain'],
                'strategy': ex['strategy']
            }, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for MonoT5 reranker')
    parser.add_argument('--domains', nargs='+', default=DOMAINS,
                        help='Domains to process')
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES,
                        help='Query strategies to use')
    parser.add_argument('--retrieval-method', type=str, default=RETRIEVAL_METHOD,
                        help='Retrieval method to use for hard negatives')
    parser.add_argument('--negative-ratio', type=int, default=NEGATIVE_RATIO,
                        help='Ratio of negatives to positives')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--corpus-level', type=str, default='passage_level',
                        choices=['passage_level', 'document_level'],
                        help='Corpus level to use')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Hardcoded paths
    PROJECT_ROOT = Path('/Users/pastil/Dev/Github/mt-rag-benchmark')
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Preparing MonoT5 Training Data")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"Strategies: {args.strategies}")
    print(f"Retrieval Method: {args.retrieval_method}")
    print(f"Negative Ratio: 1:{args.negative_ratio}")
    print(f"Output Directory: {output_dir}")
    print()
    
    all_examples = []
    
    # Process each domain
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load Qrels - hardcoded path
        qrels_path = PROJECT_ROOT / 'human' / 'retrieval_tasks' / domain / 'qrels' / 'dev.tsv'
        print(f"  Loading Qrels from: {qrels_path}")
        qrels = load_qrels(qrels_path)
        print(f"  Loaded {len(qrels)} queries with Qrels")
        
        # Load corpus - hardcoded path
        corpus_path = PROJECT_ROOT / 'corpora' / args.corpus_level / f'{domain}.jsonl'
        print(f"  Loading corpus from: {corpus_path}")
        corpus = load_corpus(corpus_path)
        print(f"  Loaded {len(corpus)} corpus documents")
        
        # Process each strategy
        for strategy in args.strategies:
            print(f"  Processing strategy: {strategy}")
            
            # Load queries - hardcoded path
            queries_path = PROJECT_ROOT / 'human' / 'retrieval_tasks' / domain / f'{domain}_{strategy}.jsonl'
            print(f"    Loading queries from: {queries_path}")
            queries = load_queries(queries_path)
            print(f"    Loaded {len(queries)} queries")
            
            # Load retrieval results - hardcoded path
            results_path = PROJECT_ROOT / 'scripts' / 'baselines' / 'retrieval_scripts' / args.retrieval_method / 'results' / f'{args.retrieval_method}_{domain}_{strategy}_evaluated.jsonl'
            print(f"    Loading retrieval results from: {results_path}")
            retrieval_results = load_retrieval_results(results_path)
            print(f"    Loaded retrieval results for {len(retrieval_results)} queries")
            
            # Create examples for each query
            domain_strategy_examples = []
            queries_with_examples = 0
            for query_id, query_text in queries.items():
                examples = create_examples(
                    query_id=query_id,
                    query_text=query_text,
                    qrels=qrels,
                    retrieval_results=retrieval_results,
                    corpus=corpus,
                    domain=domain,
                    strategy=strategy,
                    negative_ratio=args.negative_ratio
                )
                if examples:
                    queries_with_examples += 1
                    domain_strategy_examples.extend(examples)
            
            print(f"    Created {len(domain_strategy_examples)} examples from {queries_with_examples}/{len(queries)} queries")
            all_examples.extend(domain_strategy_examples)
    
    print()
    print(f"Total examples created: {len(all_examples)}")
    
    # Count positives and negatives
    n_positives = sum(1 for ex in all_examples if ex['label'] == 1)
    n_negatives = sum(1 for ex in all_examples if ex['label'] == 0)
    print(f"  Positives: {n_positives}")
    print(f"  Negatives: {n_negatives}")
    print(f"  Ratio: 1:{n_negatives/n_positives:.2f}" if n_positives > 0 else "  Ratio: N/A")
    
    # Split by task
    print()
    print("Splitting by task_id...")
    train_examples, val_examples, test_examples = split_by_task(all_examples)
    
    print()
    print(f"Split sizes:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")
    
    # Verify no leakage
    print()
    print("Verifying no data leakage...")
    if not verify_no_leakage(train_examples, val_examples, test_examples):
        print("WARNING: Data leakage detected! Please review the splits.")
        return
    
    # Save splits
    print()
    print("Saving splits...")
    save_examples(train_examples, output_dir / 'train.jsonl')
    save_examples(val_examples, output_dir / 'val.jsonl')
    save_examples(test_examples, output_dir / 'test.jsonl')
    
    # Save metadata
    metadata = {
        'total_examples': len(all_examples),
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'test_examples': len(test_examples),
        'n_positives': n_positives,
        'n_negatives': n_negatives,
        'domains': args.domains,
        'strategies': args.strategies,
        'retrieval_method': args.retrieval_method,
        'negative_ratio': args.negative_ratio,
        'seed': args.seed
    }
    
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Data preparation complete!")
    print(f"  Train: {output_dir / 'train.jsonl'} ({len(train_examples)} examples)")
    print(f"  Val: {output_dir / 'val.jsonl'} ({len(val_examples)} examples)")
    print(f"  Test: {output_dir / 'test.jsonl'} ({len(test_examples)} examples)")
    print(f"  Metadata: {output_dir / 'metadata.json'}")
    print()
    print("Note: Data files contain references only (query_id, document_id, label, etc.)")
    print("      Actual text will be loaded at training time from original data sources.")


if __name__ == '__main__':
    main()
