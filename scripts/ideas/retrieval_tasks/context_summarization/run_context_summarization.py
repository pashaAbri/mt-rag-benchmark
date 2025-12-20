#!/usr/bin/env python3
"""
Context Summarization: SELF-multi-RAG style conversation summarization for retrieval.

This script implements the summarization technique from the SELF-multi-RAG paper,
which creates a 40-50 word summary + reformulated question instead of traditional
query rewriting.

Key differences from targeted_rewrite:
1. Output format: "Summary: ... Question: ..." instead of just a rewritten query
2. Summarization includes ALL relevant context, not just semantically similar turns
3. Based on the SELF-multi-RAG paper's approach that achieved +13.5% retrieval improvement

Usage:
    # Test on first 10 conversations
    python run_context_summarization.py --max_conversations 10
    
    # Run on all conversations for specific domains
    python run_context_summarization.py --domains clapnq cloud
    
    # Run on all conversations
    python run_context_summarization.py
"""

import sys
import json
import os
import argparse
import time
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
CONVERSATIONS_FILE = project_root / "human" / "conversations" / "conversations.json"
GENERATION_TASKS_FILE = project_root / "human" / "generation_tasks" / "reference.jsonl"
OUTPUT_DIR = script_dir / "intermediate"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Domain mapping from full names to short names
DOMAIN_MAP = {
    'mt-rag-clapnq-elser-512-100-20240503': 'clapnq',
    'mt-rag-govt-elser-512-100-20240611': 'govt',
    'mt-rag-fiqa-beir-elser-512-100-20240501': 'fiqa',
    'mt-rag-ibmcloud-elser-512-100-20240502': 'cloud'
}


def build_conversation_id_mapping() -> Dict[int, str]:
    """
    Build mapping from conversation index to conversation ID (hash).
    
    The generation tasks file has the actual conversation IDs that match the qrels.
    We match conversations by their first question text and domain.
    
    Returns:
        Dictionary mapping conversation index to conversation_id
    """
    # Build mapping from (domain, first_question) -> conversation_id
    conv_map = {}
    with open(GENERATION_TASKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)
            cid = task['conversation_id']
            turn = int(task['turn'])
            collection = task.get('Collection', '')
            
            # Use first turn to identify conversation
            if turn == 1:
                first_q = task['input'][0]['text'] if task['input'] else ''
                conv_map[(collection, first_q)] = cid
    
    # Match with human conversations
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        human_convs = json.load(f)
    
    idx_to_cid = {}
    for i, conv in enumerate(human_convs):
        domain = conv.get('domain', '')
        messages = conv.get('messages', [])
        if messages and messages[0].get('speaker') == 'user':
            first_q = messages[0].get('text', '')
            key = (domain, first_q)
            if key in conv_map:
                idx_to_cid[i] = conv_map[key]
    
    return idx_to_cid


def get_llm_client():
    """Initialize Anthropic client."""
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


def load_conversations(max_conversations: Optional[int] = None, domains: Optional[List[str]] = None) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Load conversations from the human conversations file.
    
    Args:
        max_conversations: Maximum number of conversations to load (for testing)
        domains: List of domains to filter by (e.g., ['clapnq', 'cloud'])
        
    Returns:
        Tuple of (list of conversation dictionaries, index to conversation_id mapping)
    """
    # Build conversation ID mapping first
    idx_to_cid = build_conversation_id_mapping()
    
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        all_conversations = json.load(f)
    
    # Add index and conversation_id
    for i, conv in enumerate(all_conversations):
        conv['_index'] = i
        conv['_conversation_id'] = idx_to_cid.get(i, f'unknown_{i}')
        # Map domain to short name
        full_domain = conv.get('domain', '')
        conv['_domain_short'] = DOMAIN_MAP.get(full_domain, 'unknown')
    
    # Filter by domain if specified
    if domains:
        all_conversations = [c for c in all_conversations if c['_domain_short'] in domains]
    
    # Limit for testing
    if max_conversations:
        all_conversations = all_conversations[:max_conversations]
    
    return all_conversations, idx_to_cid


def get_conversation_turns(conversation: Dict) -> List[Dict]:
    """
    Extract user turns from a conversation.
    
    Returns list of dictionaries with:
    - turn_id: 1-based turn number
    - user_text: Current user question
    - history: All previous messages (for summarization)
    """
    messages = conversation.get('messages', [])
    turns = []
    
    user_turn_count = 0
    history = []
    
    for i, msg in enumerate(messages):
        if msg.get('speaker') == 'user':
            user_turn_count += 1
            
            turns.append({
                'turn_id': user_turn_count,
                'user_text': msg.get('text', ''),
                'history': list(history),  # Copy current history
                'enrichments': msg.get('enrichments', {})
            })
            
        # Add to history for next turn
        history.append({
            'speaker': msg.get('speaker', 'user'),
            'text': msg.get('text', '')
        })
    
    return turns


def summarize_conversation(
    client,
    conversation_history: List[Dict],
    current_question: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> Tuple[str, str, str]:
    """
    Generate a summary + question for the conversation.
    
    Args:
        client: Anthropic client
        conversation_history: List of previous messages
        current_question: Current user question
        model: LLM model to use
        
    Returns:
        Tuple of (summary, question, full_query)
    """
    from prompt_template import (
        get_system_prompt,
        build_user_prompt,
        parse_summary_response,
        format_retrieval_query
    )
    
    # Build prompts
    system_prompt = get_system_prompt()
    user_prompt = build_user_prompt(
        conversation_history=conversation_history,
        current_question=current_question,
        include_few_shot=True
    )
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        response = message.content[0].text.strip()
        
        # Parse response
        summary, question = parse_summary_response(response)
        
        # Format retrieval query
        full_query = format_retrieval_query(summary, question)
        
        return summary, question, full_query
        
    except Exception as e:
        print(f"  Error in summarization: {e}")
        # Fallback: return original question
        return "", current_question, current_question


def process_single_turn(
    turn_data: Dict,
    conversation: Dict,
    llm_client,
    args,
    stats_lock: threading.Lock,
    stats: Dict
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process a single conversation turn.
    
    Returns:
        Tuple of (beir_result, analysis)
    """
    conv_index = conversation['_index']
    conversation_id = conversation['_conversation_id']
    domain = conversation['_domain_short']
    turn_id = turn_data['turn_id']
    user_text = turn_data['user_text']
    history = turn_data['history']
    
    # Create task ID matching the qrels format (conversation_id<::>turn)
    task_id = f"{conversation_id}<::>{turn_id}"
    
    # Turn 1: no history, use original question
    if turn_id == 1 or not history:
        summary = ""
        question = user_text
        full_query = user_text
        method = "no_history"
        
        with stats_lock:
            stats["turn1_no_summarization"] += 1
    else:
        # Summarize conversation
        summary, question, full_query = summarize_conversation(
            client=llm_client,
            conversation_history=history,
            current_question=user_text,
            model=args.llm_model
        )
        method = "summarization"
        
        with stats_lock:
            stats["summarized"] += 1
    
    # Count words in summary
    word_count = len(summary.split()) if summary else 0
    
    # BEIR format result
    beir_result = {
        "_id": task_id,
        "text": full_query
    }
    
    # Analysis
    analysis = {
        "task_id": task_id,
        "conversation_index": conv_index,
        "domain": domain,
        "turn_id": turn_id,
        "original_query": user_text,
        "summary": summary,
        "question": question,
        "full_query": full_query,
        "method": method,
        "history_turns": len([h for h in history if h.get('speaker') == 'user']),
        "summary_word_count": word_count,
        "enrichments": turn_data.get('enrichments', {})
    }
    
    return beir_result, analysis


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


def main():
    parser = argparse.ArgumentParser(description="Context Summarization (SELF-multi-RAG style)")
    parser.add_argument("--domains", nargs="+", default=None, 
                        help="Domains to process (default: all)")
    parser.add_argument("--max_conversations", type=int, default=None,
                        help="Maximum conversations to process (for testing)")
    parser.add_argument("--llm_model", type=str, default="claude-sonnet-4-5-20250929",
                        help="LLM model for summarization")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent workers for LLM calls")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip already processed tasks")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output files (e.g., '_test')")
    args = parser.parse_args()
    
    print("="*80)
    print("Context Summarization (SELF-multi-RAG Style)")
    print("="*80)
    print(f"LLM model: {args.llm_model}")
    print(f"Concurrent workers: {args.workers}")
    print(f"Domains: {args.domains or 'all'}")
    print(f"Max conversations: {args.max_conversations or 'all'}")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM client
    print("\nInitializing LLM client...")
    llm_client = get_llm_client()
    
    # Load conversations
    print("Loading conversations...")
    conversations, idx_to_cid = load_conversations(
        max_conversations=args.max_conversations,
        domains=args.domains
    )
    print(f"Loaded {len(conversations)} conversations")
    print(f"Mapped {len(idx_to_cid)} conversation IDs")
    
    # Group by domain
    by_domain = defaultdict(list)
    for conv in conversations:
        by_domain[conv['_domain_short']].append(conv)
    
    print(f"Domains: {dict((d, len(c)) for d, c in by_domain.items())}")
    
    # Process each domain
    for domain, domain_conversations in by_domain.items():
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain}")
        print(f"{'='*60}")
        
        suffix = args.output_suffix
        output_file = OUTPUT_DIR / f"context_summary_{domain}{suffix}.jsonl"
        analysis_file = OUTPUT_DIR / f"context_summary_{domain}{suffix}_analysis.json"
        
        # Load already completed tasks
        completed_task_ids = set()
        if args.skip_existing:
            completed_task_ids = load_completed_task_ids(output_file)
            if completed_task_ids:
                print(f"Found {len(completed_task_ids)} already completed tasks")
        
        # Collect all turns to process
        pending_tasks = []
        for conv in domain_conversations:
            turns = get_conversation_turns(conv)
            for turn in turns:
                task_id = f"{conv['_conversation_id']}<::>{turn['turn_id']}"
                if task_id not in completed_task_ids:
                    pending_tasks.append((turn, conv))
        
        print(f"Processing {len(pending_tasks)} pending tasks")
        
        if not pending_tasks:
            print(f"All tasks already completed for {domain}")
            continue
        
        # Initialize results
        results = []
        analyses = []
        stats = defaultdict(int)
        stats_lock = threading.Lock()
        results_lock = threading.Lock()
        
        # Process tasks concurrently
        def process_task_wrapper(task_tuple):
            turn_data, conv = task_tuple
            try:
                result, analysis = process_single_turn(
                    turn_data=turn_data,
                    conversation=conv,
                    llm_client=llm_client,
                    args=args,
                    stats_lock=stats_lock,
                    stats=stats
                )
                return result, analysis
            except Exception as e:
                task_id = f"{conv['_conversation_id']}<::>{turn_data['turn_id']}"
                print(f"Error processing task {task_id}: {e}")
                return None, None
        
        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(process_task_wrapper, task): task
                for task in pending_tasks
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(future_to_task),
                total=len(future_to_task),
                desc=f"Summarizing {domain}"
            ):
                result, analysis = future.result()
                if result is not None:
                    with results_lock:
                        results.append(result)
                        if analysis is not None:
                            analyses.append(analysis)
        
        # Sort results by task_id for consistent output
        results.sort(key=lambda x: x['_id'])
        analyses.sort(key=lambda x: x['task_id'])
        
        # Write results
        print(f"\nWriting {len(results)} results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # Calculate summary statistics
        word_counts = [a['summary_word_count'] for a in analyses if a['summary_word_count'] > 0]
        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
        
        # Write analysis
        print(f"Writing analysis to {analysis_file}")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "llm_model": args.llm_model,
                    "max_conversations": args.max_conversations,
                    "domains": args.domains
                },
                "stats": {
                    **dict(stats),
                    "total_tasks": len(results),
                    "avg_summary_word_count": round(avg_word_count, 1),
                    "min_summary_word_count": min(word_counts) if word_counts else 0,
                    "max_summary_word_count": max(word_counts) if word_counts else 0
                },
                "analyses": analyses
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nStats for {domain}:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        print(f"  avg_summary_word_count: {avg_word_count:.1f}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

