#!/usr/bin/env python3
"""
Targeted Query Rewrite with Mixtral: Isolate the effect of context filtering.

This experiment uses the SAME LLM as the baseline (Mixtral 8x7B) to isolate
whether performance gains come from:
1. Better context filtering (the hypothesis)
2. Better LLM (the confound in the original experiment)

This script:
1. Loads conversations from cleaned_data/conversations/
2. For each query, computes semantic similarity with previous turns
3. Includes only relevant turns (above threshold) in the rewrite prompt
4. Rewrites the query using Mixtral 8x7B (same as paper baseline)
5. Saves results in BEIR format for evaluation

Usage:
    python run_targeted_rewrite.py --domains clapnq --similarity_threshold 0.3
"""

import sys
import json
import os
import argparse
import time
import requests
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

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Mixtral 8x7B configuration (same as paper baseline)
MIXTRAL_CONFIG = {
    "model_name": "mixtral_8x7b_instruct",
    "api_model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "api_endpoint": "https://api.together.xyz/v1/chat/completions",
    "max_tokens": 200,
    "temperature": 0.0,
    "top_p": 1.0,
}


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence transformer model for embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)


def get_together_api_key():
    """Get Together AI API key from environment."""
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        print("Error: TOGETHER_API_KEY not found. Please set it in .env")
        print("Get your API key from: https://api.together.xyz/")
        sys.exit(1)
    return api_key


def compute_similarity(query_emb: np.ndarray, turn_emb: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    turn_norm = turn_emb / (np.linalg.norm(turn_emb) + 1e-9)
    return float(np.dot(query_norm, turn_norm))


def select_relevant_turns(
    current_query: str,
    conversation_history: List[Dict[str, str]],
    embedding_model,
    similarity_threshold: float = 0.3,
    max_relevant_turns: int = 5,
    include_last_turn: bool = True
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Select relevant turns from conversation history based on semantic similarity.
    
    Args:
        current_query: The current user query
        conversation_history: List of previous turns (alternating user/agent)
        embedding_model: Sentence transformer model
        similarity_threshold: Minimum similarity to include a turn
        max_relevant_turns: Maximum turns to include
        include_last_turn: Always include the immediately preceding turn
        
    Returns:
        selected_turns: List of relevant turns to include
        analysis: Dictionary with similarity scores and selection details
    """
    if not conversation_history:
        return [], {"num_history_turns": 0, "selected_turns": 0}
    
    # Group history into turn pairs (user + agent response)
    turn_pairs = []
    i = 0
    while i < len(conversation_history):
        turn = {"user": "", "agent": ""}
        if conversation_history[i].get("speaker") == "user":
            turn["user"] = conversation_history[i].get("text", "")
            if i + 1 < len(conversation_history) and conversation_history[i + 1].get("speaker") == "agent":
                turn["agent"] = conversation_history[i + 1].get("text", "")
                i += 2
            else:
                i += 1
        else:
            # Agent without user (shouldn't happen often)
            turn["agent"] = conversation_history[i].get("text", "")
            i += 1
        
        if turn["user"] or turn["agent"]:
            turn_pairs.append(turn)
    
    if not turn_pairs:
        return [], {"num_history_turns": 0, "selected_turns": 0}
    
    # Create combined text for each turn pair
    turn_texts = []
    for tp in turn_pairs:
        combined = f"User: {tp['user']}"
        if tp['agent']:
            combined += f" Assistant: {tp['agent']}"
        turn_texts.append(combined)
    
    # Embed current query and all turn pairs
    query_emb = embedding_model.encode(current_query, convert_to_numpy=True)
    turn_embs = embedding_model.encode(turn_texts, convert_to_numpy=True)
    
    # Compute similarities
    similarities = []
    for idx, turn_emb in enumerate(turn_embs):
        sim = compute_similarity(query_emb, turn_emb)
        similarities.append({
            "turn_idx": idx,
            "similarity": sim,
            "user_text": turn_pairs[idx]["user"][:100],  # Truncate for logging
            "agent_text": turn_pairs[idx]["agent"][:100] if turn_pairs[idx]["agent"] else ""
        })
    
    # Sort by similarity (descending)
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    # Select turns above threshold
    selected_indices = set()
    
    # Always include last turn if requested
    if include_last_turn and turn_pairs:
        selected_indices.add(len(turn_pairs) - 1)
    
    # Add turns above threshold (up to max)
    for sim_info in similarities_sorted:
        if len(selected_indices) >= max_relevant_turns:
            break
        if sim_info["similarity"] >= similarity_threshold:
            selected_indices.add(sim_info["turn_idx"])
    
    # Convert to ordered list (maintain chronological order)
    selected_indices = sorted(selected_indices)
    
    # Build selected turns list (as original format for LLM)
    selected_turns = []
    for idx in selected_indices:
        tp = turn_pairs[idx]
        if tp["user"]:
            selected_turns.append({"speaker": "user", "text": tp["user"]})
        if tp["agent"]:
            selected_turns.append({"speaker": "agent", "text": tp["agent"]})
    
    # Analysis for debugging/understanding
    analysis = {
        "num_history_turns": len(turn_pairs),
        "selected_turns": len(selected_indices),
        "selected_indices": list(selected_indices),
        "similarities": similarities,
        "threshold": similarity_threshold,
        "above_threshold_count": sum(1 for s in similarities if s["similarity"] >= similarity_threshold)
    }
    
    return selected_turns, analysis


def rewrite_query_mixtral(
    api_key: str,
    current_query: str,
    relevant_history: List[Dict[str, str]],
    config: Dict[str, Any] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """
    Rewrite the query using Mixtral 8x7B via Together AI.
    
    Args:
        api_key: Together AI API key
        current_query: Current user query
        relevant_history: Filtered relevant turns
        config: Model configuration (defaults to MIXTRAL_CONFIG)
        max_retries: Number of retries on failure
        retry_delay: Delay between retries
        
    Returns:
        Rewritten standalone query
    """
    if config is None:
        config = MIXTRAL_CONFIG
    
    # If no history, return original query
    if not relevant_history:
        return current_query
    
    # Build history text
    history_lines = []
    for turn in relevant_history:
        speaker = turn.get("speaker", "user")
        text = turn.get("text", "")
        if speaker == "user":
            history_lines.append(f"User: {text}")
        else:
            history_lines.append(f"Assistant: {text}")
    
    history_text = "\n".join(history_lines)
    
    # Rewrite prompt (same as baseline from paper - Figure 5)
    # Note: Mixtral uses a single user message format, not system + user
    prompt = f"""You are an expert at rewriting conversational queries into standalone queries.

Given a conversation history and the user's current question, rewrite the current question into a standalone query that:
1. Contains all necessary context from the conversation to be understood independently
2. Is clear and self-contained
3. Preserves the user's original intent
4. Does NOT introduce new information not present in the conversation

If the query is already standalone and doesn't need context, return it unchanged.

Output ONLY the rewritten query, nothing else.

Conversation history:
{history_text}

Current query: {current_query}

Rewritten standalone query:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config['api_model_id'],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.get('max_tokens', 200),
        "temperature": config.get('temperature', 0.0),
        "top_p": config.get('top_p', 1.0),
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                config['api_endpoint'],
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            rewritten = result['choices'][0]['message']['content'].strip()
            
            # Clean up
            rewritten = rewritten.strip('"').strip("'")
            if rewritten.lower().startswith("rewritten"):
                rewritten = rewritten.split(":", 1)[-1].strip()
            
            return rewritten if rewritten else current_query
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"  Error rewriting query after {max_retries} attempts: {e}")
                return current_query
        except (KeyError, IndexError) as e:
            print(f"  Unexpected API response format: {e}")
            return current_query
    
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
            continue
    
    return conversations


def get_conversation_history(conversation: Dict, turn_id: int) -> List[Dict[str, str]]:
    """Get conversation history up to (but not including) the specified turn."""
    messages = conversation.get("messages", [])
    history = []
    
    user_turn_count = 0
    for msg in messages:
        if msg.get("speaker") == "user":
            user_turn_count += 1
            if user_turn_count >= turn_id:
                break  # Stop before the current turn
        
        history.append({
            "speaker": msg.get("speaker", "user"),
            "text": msg.get("text", "")
        })
    
    return history


def load_completed_task_ids(output_file: Path) -> Set[str]:
    """Load task IDs that have already been processed from JSONL file."""
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


def load_completed_analyses(analysis_file: Path) -> Dict[str, Dict]:
    """Load analyses that have already been processed."""
    completed = {}
    if not analysis_file.exists():
        return completed
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for analysis in data.get("analyses", []):
                task_id = analysis.get("task_id")
                if task_id:
                    completed[task_id] = analysis
    except Exception as e:
        print(f"Warning: Error reading analysis file: {e}")
    
    return completed


def process_single_task(
    task_data: Dict,
    conversations: Dict,
    embedding_model,
    api_key: str,
    args,
    stats_lock: threading.Lock,
    stats: Dict
) -> Tuple[Dict, Dict]:
    """
    Process a single task: select relevant turns and rewrite query.
    
    Returns:
        result: BEIR format result {"_id": task_id, "text": rewritten_query}
        analysis: Analysis dictionary with metadata
    """
    task_id = task_data.get("task_id")
    conversation_id = str(task_data.get("conversation_id"))
    turn_id = task_data.get("turn_id", 1)
    
    # Get current query
    current_query = task_data.get("user", {}).get("text", "")
    if not current_query:
        return None, None
    
    # Get conversation history
    conversation = conversations.get(conversation_id, {})
    history = get_conversation_history(conversation, turn_id)
    
    # Turn 1: no history, use original query
    if turn_id == 1 or not history:
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
        # Select relevant turns
        relevant_turns, turn_analysis = select_relevant_turns(
            current_query=current_query,
            conversation_history=history,
            embedding_model=embedding_model,
            similarity_threshold=args.similarity_threshold,
            max_relevant_turns=args.max_relevant_turns,
            include_last_turn=args.include_last_turn
        )
        
        # Rewrite with relevant context using Mixtral
        rewritten = rewrite_query_mixtral(
            api_key=api_key,
            current_query=current_query,
            relevant_history=relevant_turns,
            config=MIXTRAL_CONFIG
        )
        
        analysis = {
            "task_id": task_id,
            "turn_id": turn_id,
            "original_query": current_query,
            "rewritten_query": rewritten,
            "method": "targeted_rewrite_mixtral",
            **turn_analysis
        }
        
        with stats_lock:
            stats["rewritten"] += 1
            if turn_analysis["selected_turns"] < turn_analysis["num_history_turns"]:
                stats["turns_filtered"] += 1
    
    # Return BEIR format result and analysis
    result = {
        "_id": task_id,
        "text": f"|user|: {rewritten}"
    }
    
    return result, analysis


def main():
    parser = argparse.ArgumentParser(description="Targeted Query Rewrite with Mixtral 8x7B")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to process")
    parser.add_argument("--similarity_threshold", type=float, default=0.3, 
                        help="Minimum similarity to include a turn")
    parser.add_argument("--max_relevant_turns", type=int, default=5,
                        help="Maximum relevant turns to include")
    parser.add_argument("--include_last_turn", action="store_true", default=True,
                        help="Always include the immediately preceding turn")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embeddings")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output file already exists (entire domain)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent workers for LLM calls")
    args = parser.parse_args()
    
    print("="*80)
    print("Targeted Query Rewrite with Mixtral 8x7B")
    print("="*80)
    print(f"LLM Model: {MIXTRAL_CONFIG['model_name']} (same as paper baseline)")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Max relevant turns: {args.max_relevant_turns}")
    print(f"Include last turn: {args.include_last_turn}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Concurrent workers: {args.workers}")
    print(f"Domains: {args.domains}")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load models and API key
    print("\nLoading embedding model...")
    embedding_model = load_embedding_model(args.embedding_model)
    
    print("Getting Together AI API key...")
    api_key = get_together_api_key()
    
    # Load conversations
    print("Loading conversations...")
    conversations = load_conversations()
    print(f"Loaded {len(conversations)} conversations")
    
    # Process each domain
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain}")
        print("="*60)
        
        output_file = OUTPUT_DIR / f"targeted_rewrite_mixtral_{domain}.jsonl"
        analysis_file = OUTPUT_DIR / f"targeted_rewrite_mixtral_{domain}_analysis.json"
        
        if args.skip_existing and output_file.exists():
            print(f"Skipping {domain}: output file exists")
            continue
        
        tasks_dir = TASKS_DIR / domain
        if not tasks_dir.exists():
            print(f"Tasks directory not found: {tasks_dir}")
            continue
        
        task_files = list(tasks_dir.glob("*.json"))
        print(f"Found {len(task_files)} tasks")
        
        # Load already completed tasks
        completed_task_ids = load_completed_task_ids(output_file)
        completed_analyses = load_completed_analyses(analysis_file)
        
        if completed_task_ids:
            print(f"Found {len(completed_task_ids)} already completed tasks, will skip them")
        
        # Load all task data and filter out completed ones
        pending_tasks = []
        for task_file in task_files:
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                task_id = task_data.get("task_id")
                if task_id and task_id not in completed_task_ids:
                    pending_tasks.append(task_data)
            except Exception as e:
                print(f"Error loading {task_file.name}: {e}")
                continue
        
        print(f"Processing {len(pending_tasks)} pending tasks ({len(completed_task_ids)} already done)")
        
        if not pending_tasks:
            print(f"All tasks already completed for {domain}")
            continue
        
        # Initialize results with existing completed ones
        results = []
        analyses = list(completed_analyses.values())
        stats = defaultdict(int)
        stats_lock = threading.Lock()
        results_lock = threading.Lock()
        
        # Process tasks concurrently
        def process_task_wrapper(
            task_data,
            _conversations=conversations,
            _embedding_model=embedding_model,
            _api_key=api_key,
            _args=args,
            _stats_lock=stats_lock,
            _stats=stats
        ):
            """Wrapper to process a single task and handle exceptions."""
            try:
                result, analysis = process_single_task(
                    task_data=task_data,
                    conversations=_conversations,
                    embedding_model=_embedding_model,
                    api_key=_api_key,
                    args=_args,
                    stats_lock=_stats_lock,
                    stats=_stats
                )
                return result, analysis
            except Exception as e:
                task_id = task_data.get("task_id", "unknown")
                print(f"Error processing task {task_id}: {e}")
                return None, None
        
        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_task_wrapper, task_data): task_data
                for task_data in pending_tasks
            }
            
            # Collect results with progress bar
            for future in tqdm(
                concurrent.futures.as_completed(future_to_task),
                total=len(future_to_task),
                desc=f"Rewriting {domain}"
            ):
                result, analysis = future.result()
                if result is not None:
                    with results_lock:
                        results.append(result)
                        if analysis is not None:
                            analyses.append(analysis)
        
        # Also add back the completed results for the final output
        for task_id in completed_task_ids:
            if task_id in completed_analyses:
                analysis = completed_analyses[task_id]
                rewritten = analysis.get("rewritten_query", "")
                if rewritten:
                    results.append({
                        "_id": task_id,
                        "text": f"|user|: {rewritten}"
                    })
        
        # Write results (overwrite with complete set)
        print(f"\nWriting {len(results)} results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # Write analysis (complete set)
        print(f"Writing analysis to {analysis_file}")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "similarity_threshold": args.similarity_threshold,
                    "max_relevant_turns": args.max_relevant_turns,
                    "include_last_turn": args.include_last_turn,
                    "embedding_model": args.embedding_model,
                    "llm_model": MIXTRAL_CONFIG['model_name'],
                    "llm_api_model_id": MIXTRAL_CONFIG['api_model_id']
                },
                "stats": dict(stats),
                "analyses": analyses
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nStats for {domain} (this run):")
        for key, val in stats.items():
            print(f"  {key}: {val}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

