#!/usr/bin/env python3
"""
Baseline Pipeline: Rewrite with FULL History → Retrieve → Generate
Using Claude Sonnet for LLM calls.

This script establishes a BASELINE for Claude Sonnet using FULL conversation history
(no filtering). This allows fair comparison with the targeted rewrite approach.

Pipeline for each turn:
  1. Rewrite query using FULL conversation history (Claude Sonnet)
  2. Retrieve documents using rewritten query (ELSER)
  3. Generate response from retrieved documents (Claude Sonnet)
  4. Add generated response to context for next turn

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
    SONNET_CONFIG,
    ELSER_INDEX_MAPPING,
    COLLECTION_MAPPING,
    OUTPUT_DIR,
    # Model loading
    get_anthropic_api_key,
    get_elasticsearch_client,
    # LLM calls (FULL history, no filtering)
    rewrite_query_sonnet,
    generate_response_sonnet,
    # Retrieval
    retrieve_documents_elser,
    # Data loading
    load_tasks_by_conversation,
)


def load_completed_conversations(output_file: Path, analysis_file: Path) -> Tuple[Set[str], List[Dict], List[Dict]]:
    """Load already completed results to support resuming."""
    existing_results = []
    existing_analyses = []
    completed_conv_ids = set()
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    existing_results.append(result)
                    task_id = result.get("task_id", "")
                    if "<::>" in task_id:
                        conv_id = task_id.split("<::>")[0]
                        completed_conv_ids.add(conv_id)
    
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
    """Save results incrementally."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": config,
            "stats": dict(stats),
            "analyses": analyses
        }, f, indent=2, ensure_ascii=False)


def process_conversation(
    tasks: List[Dict],
    anthropic_api_key: str,
    es_client,
    index_name: str,
    collection_name: str,
    top_k: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process all turns in a conversation with FULL history (no filtering):
    rewrite → retrieve → generate
    """
    retrieval_results = []
    analyses = []
    
    # Build context progressively with GENERATED responses
    progressive_context = []
    
    for task in tasks:
        task_id = task.get("task_id")
        turn_id = task.get("turn_id", 1)
        current_query = task.get("user", {}).get("text", "")
        
        if not current_query:
            continue
        
        # --- Step 1: Rewrite query using FULL history (no filtering) ---
        if turn_id == 1 or not progressive_context:
            rewritten = current_query
            method = "no_history"
            num_history_turns = 0
        else:
            # Use ALL previous turns (no filtering)
            rewritten = rewrite_query_sonnet(
                api_key=anthropic_api_key,
                current_query=current_query,
                full_history=progressive_context  # FULL history
            )
            method = "full_history_rewrite"
            num_history_turns = len(progressive_context) // 2  # Each turn = user + agent
        
        # --- Step 2: Retrieve documents ---
        contexts = retrieve_documents_elser(
            es=es_client,
            index_name=index_name,
            query_text=rewritten,
            top_k=top_k
        )
        
        # --- Step 3: Generate response ---
        generated_response = generate_response_sonnet(
            api_key=anthropic_api_key,
            current_query=rewritten,
            contexts=contexts,
            conversation_history=progressive_context[-4:] if progressive_context else []
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
            "generated_response": generated_response,
            "method": method,
            "num_history_turns": num_history_turns
        })
        
        # --- Update progressive context with REWRITTEN query and GENERATED response ---
        progressive_context.append({"speaker": "user", "text": rewritten})
        progressive_context.append({"speaker": "agent", "text": generated_response})
    
    return retrieval_results, analyses


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Pipeline: Rewrite with FULL History (Claude Sonnet)"
    )
    parser.add_argument("--domains", nargs="+", default=DOMAINS, 
                        help="Domains to process")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of documents to retrieve")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from scratch (ignore existing results)")
    args = parser.parse_args()
    
    print("="*80)
    print("Baseline Pipeline: Rewrite with FULL History → Retrieve → Generate")
    print("="*80)
    print(f"LLM Model: {SONNET_CONFIG['model_name']} (Claude Sonnet)")
    print("Context Strategy: FULL conversation history (NO filtering)")
    print(f"Top-k retrieval: {args.top_k}")
    print(f"Domains: {args.domains}")
    print("="*80)
    print("\n⚠️  This uses GENERATED responses (not ground truth) for context!")
    print("    This establishes a fair baseline for Sonnet.\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Getting Anthropic API key...")
    anthropic_api_key = get_anthropic_api_key()
    
    print("Connecting to Elasticsearch...")
    es_client = get_elasticsearch_client()
    print("✓ Connected to Elasticsearch\n")
    
    config = {
        "context_strategy": "full_history",
        "llm_model": SONNET_CONFIG['model_name'],
        "top_k": args.top_k,
        "processing_mode": "baseline_full_pipeline"
    }
    
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain}")
        print("="*60)
        
        output_file = OUTPUT_DIR / f"baseline_rewrite_{domain}_elser.jsonl"
        analysis_file = OUTPUT_DIR / f"baseline_rewrite_{domain}_elser_analysis.json"
        
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
        
        # Initialize stats
        stats = defaultdict(int)
        for a in all_analyses:
            if a.get("method") == "no_history":
                stats["turn1_no_rewrite"] += 1
            else:
                stats["full_history_rewrite"] += 1
        
        # Process each remaining conversation
        for conv_id in tqdm(remaining_convs, desc=f"Processing {domain}"):
            tasks = remaining_convs[conv_id]
            
            results, analyses = process_conversation(
                tasks=tasks,
                anthropic_api_key=anthropic_api_key,
                es_client=es_client,
                index_name=index_name,
                collection_name=collection_name,
                top_k=args.top_k
            )
            
            all_results.extend(results)
            all_analyses.extend(analyses)
            
            # Update stats
            for a in analyses:
                if a.get("method") == "no_history":
                    stats["turn1_no_rewrite"] += 1
                else:
                    stats["full_history_rewrite"] += 1
            
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
    print("Done! Baseline retrieval results saved for evaluation.")
    print("="*80)
    print("\nTo evaluate, run:")
    print("  python evaluate_and_compare.py --run_eval --retriever elser")


if __name__ == "__main__":
    main()

