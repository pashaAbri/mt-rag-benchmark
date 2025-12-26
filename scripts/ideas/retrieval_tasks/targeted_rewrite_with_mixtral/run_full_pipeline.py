#!/usr/bin/env python3
"""
Full Pipeline: Targeted Rewrite → Retrieve → Generate (Sequential Turn Processing)

This script simulates a REAL conversational RAG system:
  - For each turn, we use only GENERATED responses from previous turns (not ground truth)
  - This properly tests the targeted rewrite approach in a realistic setting
  - Saves incrementally after each conversation (supports resuming)

Pipeline for each turn:
  1. Filter conversation history (semantic similarity to current query)
  2. Rewrite query using filtered context (Mixtral)
  3. Retrieve documents using rewritten query (ELSER)
  4. Generate response from retrieved documents (Mixtral)
  5. Add generated response to context for next turn

Usage:
    python run_full_pipeline.py --domains clapnq cloud fiqa govt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

from utils import (
    # Constants
    DOMAINS,
    MIXTRAL_CONFIG,
    ELSER_INDEX_MAPPING,
    COLLECTION_MAPPING,
    OUTPUT_DIR,
    # Model loading
    load_embedding_model,
    get_together_api_key,
    get_elasticsearch_client,
    # Context filtering
    select_relevant_turns,
    # LLM calls
    rewrite_query_mixtral,
    generate_response_mixtral,
    # Retrieval
    retrieve_documents_elser,
    # Data loading
    load_tasks_by_conversation,
)


def load_completed_conversations(output_file: Path, analysis_file: Path) -> Tuple[Set[str], List[Dict], List[Dict]]:
    """
    Load already completed results to support resuming.
    
    Returns:
        completed_conv_ids: Set of conversation IDs that have been fully processed
        existing_results: List of existing retrieval results
        existing_analyses: List of existing analyses
    """
    existing_results = []
    existing_analyses = []
    completed_conv_ids = set()
    
    # Load existing retrieval results
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    existing_results.append(result)
                    # Extract conversation ID from task_id (format: "conv_id<::>turn_id")
                    task_id = result.get("task_id", "")
                    if "<::>" in task_id:
                        conv_id = task_id.split("<::>")[0]
                        completed_conv_ids.add(conv_id)
    
    # Load existing analyses
    if analysis_file.exists():
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_analyses = data.get("analyses", [])
    
    return completed_conv_ids, existing_results, existing_analyses


def save_results_incremental(
    results: List[Dict],
    analyses: List[Dict],
    stats: Dict,
    output_file: Path,
    analysis_file: Path,
    config: Dict
):
    """Save results incrementally (overwrites with current state)."""
    # Save retrieval results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Save analysis
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": config,
            "stats": dict(stats),
            "analyses": analyses
        }, f, indent=2, ensure_ascii=False)


def process_conversation(
    tasks: List[Dict],
    embedding_model,
    together_api_key: str,
    es_client,
    index_name: str,
    collection_name: str,
    similarity_threshold: float,
    include_last_turn: bool,
    top_k: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process all turns in a conversation with full pipeline:
    filter → rewrite → retrieve → generate
    """
    retrieval_results = []
    analyses = []
    
    # Build context progressively with GENERATED responses (not ground truth!)
    progressive_context = []
    
    for task in tasks:
        task_id = task.get("task_id")
        turn_id = task.get("turn_id", 1)
        current_query = task.get("user", {}).get("text", "")
        
        if not current_query:
            continue
        
        # --- Step 1: Filter context and rewrite query ---
        if turn_id == 1 or not progressive_context:
            rewritten = current_query
            filter_analysis = {"method": "no_history", "num_history_turns": 0, "selected_turns": 0}
        else:
            relevant_turns, turn_analysis = select_relevant_turns(
                current_query=current_query,
                conversation_history=progressive_context,
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
                include_last_turn=include_last_turn
            )
            
            rewritten = rewrite_query_mixtral(
                api_key=together_api_key,
                current_query=current_query,
                relevant_history=relevant_turns
            )
            
            filter_analysis = {
                "method": "targeted_rewrite",
                "num_history_turns": turn_analysis.get("num_history_turns", 0),
                "selected_turns": turn_analysis.get("selected_turns", 0)
            }
        
        # --- Step 2: Retrieve documents ---
        contexts = retrieve_documents_elser(
            es=es_client,
            index_name=index_name,
            query_text=rewritten,
            top_k=top_k
        )
        
        # --- Step 3: Generate response ---
        generated_response = generate_response_mixtral(
            api_key=together_api_key,
            current_query=rewritten,
            contexts=contexts,
            conversation_history=progressive_context[-4:] if progressive_context else []  # Last 2 turns
        )
        
        # --- Save retrieval result ---
        retrieval_results.append({
            "task_id": task_id,
            "Collection": collection_name,
            "contexts": contexts
        })
        
        # --- Save analysis ---
        analyses.append({
            "task_id": task_id,
            "turn_id": turn_id,
            "original_query": current_query,
            "rewritten_query": rewritten,
            "generated_response": generated_response[:200],  # Truncate for analysis
            **filter_analysis
        })
        
        # --- Update progressive context with REWRITTEN query and GENERATED response ---
        progressive_context.append({"speaker": "user", "text": rewritten})
        progressive_context.append({"speaker": "agent", "text": generated_response})
    
    return retrieval_results, analyses


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Targeted Rewrite → Retrieve → Generate"
    )
    parser.add_argument("--domains", nargs="+", default=DOMAINS, 
                        help="Domains to process")
    parser.add_argument("--similarity_threshold", type=float, default=0.3, 
                        help="Minimum similarity to include a turn")
    parser.add_argument("--include_last_turn", action="store_true", default=True,
                        help="Always include the immediately preceding turn")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embeddings")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of documents to retrieve")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from scratch (ignore existing results)")
    args = parser.parse_args()
    
    print("="*80)
    print("Full Pipeline: Targeted Rewrite → Retrieve → Generate")
    print("="*80)
    print(f"LLM Model: {MIXTRAL_CONFIG['model_name']} (same as paper baseline)")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Include last turn: {args.include_last_turn}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Top-k retrieval: {args.top_k}")
    print(f"Domains: {args.domains}")
    print("="*80)
    print("\n⚠️  This uses GENERATED responses (not ground truth) for context!")
    print("    This simulates a real conversational RAG system.")
    print("    Results are saved after each conversation (supports resuming).\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading embedding model...")
    embedding_model = load_embedding_model(args.embedding_model)
    
    print("Getting Together AI API key...")
    together_api_key = get_together_api_key()
    
    print("Connecting to Elasticsearch...")
    es_client = get_elasticsearch_client()
    print("✓ Connected to Elasticsearch\n")
    
    config = {
        "similarity_threshold": args.similarity_threshold,
        "include_last_turn": args.include_last_turn,
        "embedding_model": args.embedding_model,
        "llm_model": MIXTRAL_CONFIG['model_name'],
        "top_k": args.top_k,
        "processing_mode": "full_pipeline"
    }
    
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain}")
        print("="*60)
        
        output_file = OUTPUT_DIR / f"targeted_rewrite_{domain}_elser.jsonl"
        analysis_file = OUTPUT_DIR / f"targeted_rewrite_{domain}_elser_analysis.json"
        
        index_name = ELSER_INDEX_MAPPING.get(domain)
        collection_name = COLLECTION_MAPPING.get(domain)
        
        if not index_name:
            print(f"Unknown domain: {domain}")
            continue
        
        # Load existing results (for resuming)
        if args.restart:
            completed_conv_ids = set()
            all_results = []
            all_analyses = []
            print("Starting fresh (--restart flag)")
        else:
            completed_conv_ids, all_results, all_analyses = load_completed_conversations(
                output_file, analysis_file
            )
            if completed_conv_ids:
                print(f"Resuming: {len(completed_conv_ids)} conversations already completed")
        
        # Load tasks grouped by conversation
        tasks_by_conv = load_tasks_by_conversation(domain)
        total_tasks = sum(len(tasks) for tasks in tasks_by_conv.values())
        
        # Filter out already completed conversations
        remaining_convs = {k: v for k, v in tasks_by_conv.items() if k not in completed_conv_ids}
        remaining_tasks = sum(len(tasks) for tasks in remaining_convs.values())
        
        print(f"Found {len(tasks_by_conv)} conversations with {total_tasks} total tasks")
        print(f"Remaining: {len(remaining_convs)} conversations with {remaining_tasks} tasks")
        
        if not remaining_convs:
            print("All conversations already processed!")
            continue
        
        # Initialize stats from existing analyses
        stats = defaultdict(int)
        for a in all_analyses:
            if a.get("method") == "no_history":
                stats["turn1_no_rewrite"] += 1
            else:
                stats["rewritten"] += 1
                if a.get("selected_turns", 0) < a.get("num_history_turns", 0):
                    stats["turns_filtered"] += 1
        
        # Process each remaining conversation
        for conv_id in tqdm(remaining_convs, desc=f"Processing {domain}"):
            tasks = remaining_convs[conv_id]
            
            results, analyses = process_conversation(
                tasks=tasks,
                embedding_model=embedding_model,
                together_api_key=together_api_key,
                es_client=es_client,
                index_name=index_name,
                collection_name=collection_name,
                similarity_threshold=args.similarity_threshold,
                include_last_turn=args.include_last_turn,
                top_k=args.top_k
            )
            
            all_results.extend(results)
            all_analyses.extend(analyses)
            
            # Update stats
            for a in analyses:
                if a.get("method") == "no_history":
                    stats["turn1_no_rewrite"] += 1
                else:
                    stats["rewritten"] += 1
                    if a.get("selected_turns", 0) < a.get("num_history_turns", 0):
                        stats["turns_filtered"] += 1
            
            # Save after each conversation
            save_results_incremental(
                results=all_results,
                analyses=all_analyses,
                stats=stats,
                output_file=output_file,
                analysis_file=analysis_file,
                config=config
            )
        
        print(f"\n✓ Saved {len(all_results)} retrieval results to {output_file}")
        print(f"✓ Saved analysis to {analysis_file}")
        
        print(f"\nStats for {domain}:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
    
    print("\n" + "="*80)
    print("Done! Retrieval results saved for evaluation.")
    print("="*80)
    print("\nTo evaluate, run:")
    print("  python evaluate_and_compare.py --run_eval --retriever elser")


if __name__ == "__main__":
    main()
