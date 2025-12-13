#!/usr/bin/env python3
"""
DH-RAG Query Rewrite: Use Dynamic Historical RAG for history selection.

This script:
1. Loads conversations from cleaned_data/conversations/
2. For each query, uses DH-RAG to select relevant history (clustering + hierarchical matching)
3. Rewrites the query using an LLM with selected history
4. Saves results in BEIR format for evaluation

Based on: DH-RAG paper (Dynamic Historical RAG)

Usage:
    python run_dh_rag_rewrite.py --domains clapnq --alpha 0.6
"""

import sys
import json
import os
import argparse
import time
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import threading

# Load environment variables
load_dotenv()

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
CONVERSATIONS_DIR = project_root / "cleaned_data" / "conversations"
TASKS_DIR = project_root / "cleaned_data" / "tasks"
OUTPUT_DIR = script_dir / "intermediate"
RESULTS_DIR = script_dir / "retrieval_results"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Import DH-RAG
from dh_rag_lite import DHRAG


def get_llm_client():
    """Initialize Anthropic client for rewriting."""
    try:
        import anthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found. Please set it in .env")
            sys.exit(1)
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("Error: anthropic not installed. Install with: pip install anthropic")
        sys.exit(1)


def rewrite_query(
    client,
    current_query: str,
    relevant_history: List[Dict[str, str]],
    model: str = "claude-sonnet-4-5-20250929"
) -> str:
    """
    Rewrite the query using only the relevant history.
    """
    if not relevant_history:
        return current_query
    
    # Build history text from DH-RAG results
    history_lines = []
    for turn in relevant_history:
        role = turn.get("role", "user")
        if role == "user":
            history_lines.append(f"User: {turn.get('query', '')}")
        else:
            history_lines.append(f"Assistant: {turn.get('response', '')}")
    
    history_text = "\n".join(history_lines)
    
    system_prompt = """You are an expert at rewriting conversational queries into standalone queries.

Given a conversation history and the user's current question, rewrite the current question into a standalone query that:
1. Contains all necessary context from the conversation to be understood independently
2. Is clear and self-contained
3. Preserves the user's original intent
4. Does NOT introduce new information not present in the conversation

If the query is already standalone and doesn't need context, return it unchanged.

Output ONLY the rewritten query, nothing else."""

    user_message = f"""Conversation history:
{history_text}

Current query: {current_query}

Rewritten standalone query:"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        rewritten = message.content[0].text.strip()
        rewritten = rewritten.strip('"').strip("'")
        if rewritten.lower().startswith("rewritten"):
            rewritten = rewritten.split(":", 1)[-1].strip()
        return rewritten if rewritten else current_query
    except Exception as e:
        print(f"  Error rewriting query: {e}")
        return current_query


def load_conversations() -> Dict[str, Dict]:
    """Load all conversations from cleaned_data/conversations/."""
    conversations = {}
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            conv_id = conv_file.stem
            conversations[conv_id] = data
        except Exception as e:
            print(f"Error loading {conv_file}: {e}")
    return conversations


def build_dh_rag_for_conversation(
    conversation: Dict,
    turn_id: int,
    args,
    shared_embedding_model=None
) -> Tuple[DHRAG, List[Dict]]:
    """
    Build DH-RAG database from conversation history up to the current turn.
    
    Returns:
        dh_rag: Populated DH-RAG instance
        history_turns: List of turn dicts for reference
    """
    dh_rag = DHRAG(
        embedding_model=args.embedding_model,
        alpha=args.alpha,
        max_clusters=args.max_clusters,
        chain_similarity_threshold=args.chain_threshold,
        shared_embedding_model=shared_embedding_model
    )
    
    messages = conversation.get("messages", [])
    history_turns = []
    
    # Process messages into (query, response) pairs
    user_turn_count = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("speaker") == "user":
            user_turn_count += 1
            if user_turn_count >= turn_id:
                break  # Stop before current turn
            
            query = msg.get("text", "")
            response = ""
            
            # Get agent response if exists
            if i + 1 < len(messages) and messages[i + 1].get("speaker") == "agent":
                response = messages[i + 1].get("text", "")
                i += 1
            
            if query:
                dh_rag.add_interaction(query, response)
                history_turns.append({"query": query, "response": response})
        i += 1
    
    return dh_rag, history_turns


def select_history_with_dh_rag(
    dh_rag: DHRAG,
    current_query: str,
    top_k: int = 3
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Use DH-RAG to select relevant history for the current query.
    
    Returns:
        selected_turns: List of relevant (query, response) pairs
        analysis: Analysis dictionary with scores and metadata
    """
    if len(dh_rag.history) == 0:
        return [], {"method": "no_history", "selected": 0}
    
    # Retrieve using DH-RAG
    results = dh_rag.retrieve(current_query, top_k=top_k)
    
    # Convert to turn format for rewriting
    selected_turns = []
    for r in results:
        # Add user query
        selected_turns.append({
            "role": "user",
            "query": r["query"],
            "response": r["response"]
        })
    
    # Sort by turn_id to maintain chronological order
    selected_turns.sort(key=lambda x: results[[r['query'] for r in results].index(x['query'])].get('triple_id', 0))
    
    # Build analysis (convert numpy floats to Python floats for JSON serialization)
    analysis = {
        "method": "dh_rag",
        "num_history_turns": len(dh_rag.history),
        "selected_turns": len(results),
        "num_clusters": len(dh_rag.clusters),
        "num_chains": dh_rag.get_chain_stats()["num_chains"],
        "retrieved_details": [
            {
                "turn_id": int(r["triple_id"]),
                "score": float(r["score"]),
                "relevance": float(r["relevance"]),
                "recency": float(r["recency"]),
                "cluster_match": bool(r["matched_cluster"]),
                "summary_match": bool(r["matched_summary"]),
                "chain_match": bool(r["in_active_chain"])
            }
            for r in results
        ]
    }
    
    return selected_turns, analysis


def load_completed_task_ids(output_file: Path) -> Set[str]:
    """Load task IDs that have already been processed."""
    completed = set()
    if not output_file.exists():
        return completed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    task_id = data.get("_id")
                    if task_id:
                        completed.add(task_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Error reading output file: {e}")
    return completed


def process_single_task(
    task_data: Dict,
    conversations: Dict,
    llm_client,
    args,
    stats_lock: threading.Lock,
    stats: Dict,
    shared_embedding_model=None
) -> Tuple[Dict, Dict]:
    """Process a single task using DH-RAG for history selection."""
    task_id = task_data.get("task_id")
    conversation_id = str(task_data.get("conversation_id"))
    turn_id = task_data.get("turn_id", 1)
    
    # Get current query
    current_query = task_data.get("user", {}).get("text", "")
    if not current_query:
        return None, None
    
    # Get conversation
    conversation = conversations.get(conversation_id, {})
    
    # Turn 1: no history
    if turn_id == 1:
        rewritten = current_query
        analysis = {
            "task_id": task_id,
            "turn_id": turn_id,
            "original_query": current_query,
            "rewritten_query": rewritten,
            "method": "no_history",
            "num_history_turns": 0,
            "selected_turns": 0
        }
        with stats_lock:
            stats["turn1_no_rewrite"] += 1
    else:
        # Build DH-RAG from history (with shared embedding model to avoid threading issues)
        dh_rag, _ = build_dh_rag_for_conversation(
            conversation, turn_id, args, shared_embedding_model=shared_embedding_model
        )
        
        # Select relevant history using DH-RAG
        selected_turns, turn_analysis = select_history_with_dh_rag(
            dh_rag, current_query, top_k=args.top_k
        )
        
        # Rewrite with selected history
        rewritten = rewrite_query(
            client=llm_client,
            current_query=current_query,
            relevant_history=selected_turns,
            model=args.llm_model
        )
        
        analysis = {
            "task_id": task_id,
            "turn_id": turn_id,
            "original_query": current_query,
            "rewritten_query": rewritten,
            **turn_analysis
        }
        
        with stats_lock:
            stats["rewritten"] += 1
            if turn_analysis.get("selected_turns", 0) < turn_analysis.get("num_history_turns", 0):
                stats["turns_filtered"] += 1
    
    result = {
        "_id": task_id,
        "text": f"|user|: {rewritten}"
    }
    
    return result, analysis


def main():
    parser = argparse.ArgumentParser(description="DH-RAG Query Rewrite")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to process")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Weight for relevance vs recency (0-1). Higher = more semantic")
    parser.add_argument("--max_clusters", type=int, default=5,
                        help="Maximum clusters for topic detection")
    parser.add_argument("--chain_threshold", type=float, default=0.4,
                        help="Similarity threshold for Chain-of-Thought tracking")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of history turns to retrieve")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embeddings")
    parser.add_argument("--llm_model", type=str, default="claude-sonnet-4-5-20250929",
                        help="LLM model for rewriting")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output file already exists")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent workers for LLM calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("DH-RAG Query Rewrite")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Alpha (relevance weight): {args.alpha}")
    print(f"  Max clusters: {args.max_clusters}")
    print(f"  Chain threshold: {args.chain_threshold}")
    print(f"  Top-K retrieval: {args.top_k}")
    print(f"  Domains: {args.domains}")
    print()
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load conversations
    print("Loading conversations...")
    conversations = load_conversations()
    print(f"Loaded {len(conversations)} conversations")
    
    # Initialize LLM client
    print("Initializing LLM client...")
    llm_client = get_llm_client()
    
    # Pre-load shared embedding model to avoid threading issues
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    shared_embedding_model = SentenceTransformer(args.embedding_model)
    print(f"Loaded {args.embedding_model}")
    
    # Process each domain
    for domain in args.domains:
        print(f"\n{'=' * 80}")
        print(f"Processing domain: {domain}")
        print("=" * 80)
        
        # Output files
        output_file = OUTPUT_DIR / f"dh_rag_{domain}.jsonl"
        analysis_file = OUTPUT_DIR / f"dh_rag_{domain}_analysis.json"
        
        if args.skip_existing and output_file.exists():
            print(f"Skipping {domain} - output exists")
            continue
        
        # Load existing progress if resuming
        completed_ids = set()
        existing_results = []
        existing_analyses = []
        
        if args.resume and output_file.exists():
            completed_ids = load_completed_task_ids(output_file)
            print(f"Resuming from {len(completed_ids)} completed tasks")
            
            # Load existing results
            with open(output_file, 'r') as f:
                for line in f:
                    existing_results.append(json.loads(line.strip()))
            
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                    existing_analyses = data.get("analyses", [])
        
        # Load tasks for this domain
        task_files = list((TASKS_DIR / domain).glob("*.json"))
        print(f"Found {len(task_files)} tasks")
        
        # Load all tasks
        tasks = []
        for tf in task_files:
            try:
                with open(tf, 'r') as f:
                    task = json.load(f)
                if task.get("task_id") not in completed_ids:
                    tasks.append(task)
            except Exception as e:
                print(f"Error loading {tf}: {e}")
        
        print(f"Processing {len(tasks)} remaining tasks")
        
        # Stats
        stats = {"rewritten": 0, "turns_filtered": 0, "turn1_no_rewrite": 0}
        stats_lock = threading.Lock()
        
        results = list(existing_results)
        analyses = list(existing_analyses)
        
        # Process tasks with progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_task,
                    task, conversations, llm_client, args, stats_lock, stats,
                    shared_embedding_model=shared_embedding_model
                ): task
                for task in tasks
            }
            
            with tqdm(total=len(futures), desc=f"DH-RAG {domain}") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result, analysis = future.result()
                        if result and analysis:
                            results.append(result)
                            analyses.append(analysis)
                    except Exception as e:
                        print(f"Error processing task: {e}")
                    pbar.update(1)
        
        # Save results (BEIR format)
        print(f"\nSaving {len(results)} results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        # Save analysis
        analysis_data = {
            "config": {
                "alpha": args.alpha,
                "max_clusters": args.max_clusters,
                "chain_threshold": args.chain_threshold,
                "top_k": args.top_k,
                "embedding_model": args.embedding_model,
                "llm_model": args.llm_model
            },
            "stats": stats,
            "analyses": analyses
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\nStats for {domain}:")
        print(f"  Rewritten: {stats['rewritten']}")
        print(f"  Turns filtered: {stats['turns_filtered']}")
        print(f"  Turn 1 (no rewrite): {stats['turn1_no_rewrite']}")
    
    print("\n" + "=" * 80)
    print("DH-RAG rewriting complete!")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nNext step: Run retrieval with run_retrieval.py")


if __name__ == "__main__":
    main()

