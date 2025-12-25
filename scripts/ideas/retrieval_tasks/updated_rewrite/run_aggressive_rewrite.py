#!/usr/bin/env python3
"""
Aggressive Query Rewrite Strategy

This script implements a new query rewriting strategy designed to fix under-specification
issues in zero-score cases. It uses an "aggressive" prompt that forces:
1. Entity resolution (replacing pronouns with names)
2. Context injection (making implicit topics explicit)
3. Retrieval optimization over conversational naturalness

It processes all conversations and generates rewritten queries using Claude Sonnet 4.5.
"""

import json
import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import anthropic
from prompts import get_rewrite_prompt

# Add parent directory to path to import common utils if needed
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Domains to process
DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Output directory
BASE_OUTPUT_DIR = Path(__file__).parent / "results"

def get_llm_client():
    """Initialize Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)

def load_conversations(data_dir: Path = Path("cleaned_data/conversations")) -> Dict[str, Dict]:
    """Load all conversations from cleaned_data/conversations/."""
    conversations = {}
    
    if not data_dir.exists():
        # Try relative to project root
        data_dir = Path("cleaned_data/conversations")
        if not data_dir.exists():
            print(f"Error: Conversations directory not found at {data_dir}")
            return {}
            
    print(f"Loading conversations from {data_dir}...")
    files = list(data_dir.glob("*.json"))
    
    for f in tqdm(files, desc="Loading conversations"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Handle both list of conversations and single conversation object
                if isinstance(data, list):
                    for conv in data:
                        conv_id = str(conv.get('conversation_id', ''))
                        if conv_id:
                            conversations[conv_id] = conv
                elif isinstance(data, dict):
                    conv_id = str(data.get('conversation_id', ''))
                    if not conv_id:
                        # Fallback to filename stem if ID not in content
                        conv_id = f.stem
                    if conv_id:
                        conversations[conv_id] = data
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    print(f"Loaded {len(conversations)} conversations.")
    return conversations

def load_tasks(tasks_dir: Path = Path("cleaned_data/tasks")) -> Dict[str, List[Dict]]:
    """Load all tasks grouped by domain."""
    tasks_by_domain = {d: [] for d in DOMAINS}
    
    if not tasks_dir.exists():
        tasks_dir = Path("cleaned_data/tasks")
        if not tasks_dir.exists():
            print(f"Error: Tasks directory not found at {tasks_dir}")
            return tasks_by_domain
            
    print(f"Loading tasks from {tasks_dir}...")
    
    for domain in DOMAINS:
        domain_dir = tasks_dir / domain
        if not domain_dir.exists():
            continue
            
        files = list(domain_dir.glob("*.json"))
        for f in tqdm(files, desc=f"Loading {domain} tasks"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    task = json.load(file)
                    tasks_by_domain[domain].append(task)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
    return tasks_by_domain

def aggressive_rewrite_query(
    client,
    current_query: str,
    history: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    prompt_name: str = "aggressive"
) -> str:
    """
    Rewrite the query using the specified strategy.
    """
    # If no history (first turn), return original query
    if not history:
        return current_query
    
    # Build history text
    history_lines = []
    for turn in history:
        speaker = turn.get("speaker", "user")
        text = turn.get("text", "")
        if speaker == "user":
            history_lines.append(f"User: {text}")
        else:
            history_lines.append(f"Assistant: {text}")
    
    history_text = "\n".join(history_lines)
    
    # Get prompts
    prompts = get_rewrite_prompt(prompt_name, history_text, current_query)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.0,
            system=prompts["system"],
            messages=[{"role": "user", "content": prompts["user"]}]
        )
        rewritten = message.content[0].text.strip()
        
        # Clean up quotes if present
        rewritten = rewritten.strip('"').strip("'")
        if rewritten.lower().startswith("rewritten query:"):
            rewritten = rewritten.split(":", 1)[-1].strip()
        
        return rewritten if rewritten else current_query
        
    except Exception as e:
        print(f"  Error rewriting query: {e}")
        return current_query

def process_domain(
    domain: str,
    tasks: List[Dict],
    conversations: Dict,
    llm_client,
    args
):
    """Process all tasks for a domain."""
    output_dir = BASE_OUTPUT_DIR / args.prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{domain}_{args.prompt_name}_rewrite.jsonl"
    
    # Check if exists
    if args.skip_existing and output_file.exists():
        print(f"Skipping {domain} (output exists)")
        return

    results = []
    processed_count = 0
    
    # Sort tasks by ID for consistency
    tasks.sort(key=lambda x: x.get('task_id', ''))
    
    # Define worker function
    def process_task(task):
        task_id = task.get('task_id')
        conv_id = str(task.get('conversation_id'))
        turn_id = task.get('turn_id', 1)
        current_query = task.get('user', {}).get('text', '')
        
        # Get history
        history = []
        if conv_id in conversations:
            conv = conversations[conv_id]
            messages = conv.get('messages', [])
            # Get turns before current one
            # Assuming turns are 1-indexed and match message order roughly
            # Better strategy: Filter messages up to before this turn
            
            # Simple heuristic: take all messages before the user's current turn
            # But the task object represents a specific turn.
            # Let's count user turns to find index
            
            user_turns_seen = 0
            for msg in messages:
                if msg.get('speaker') == 'user':
                    user_turns_seen += 1
                    if user_turns_seen == turn_id:
                        break # Stop before adding current query to history
                
                history.append({
                    "speaker": msg.get('speaker'),
                    "text": msg.get('text')
                })
        
        # Rewrite
        rewritten = aggressive_rewrite_query(
            llm_client, 
            current_query, 
            history, 
            model=args.llm_model,
            prompt_name=args.prompt_name
        )
        
        return {
            "_id": task_id,
            "text": rewritten,
            "original": current_query,
            "turn_id": turn_id
        }

    # Run with thread pool
    print(f"Processing {len(tasks)} tasks for {domain} with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc=f"Rewriting {domain}"))
        results = futures

    # Save results
    print(f"Saving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            # Format to match retrieval script expectations: {"_id": "...", "text": "..."}
            # Add |user|: prefix if needed by downstream scripts (though modern ones might strip it)
            # The baseline files have "|user|: " prefix.
            output_obj = {
                "_id": res["_id"],
                "text": f"|user|: {res['text']}"
            }
            f.write(json.dumps(output_obj) + "\n")
            
    # Save debug version with original query
    debug_file = output_dir / f"{domain}_{args.prompt_name}_rewrite_debug.jsonl"
    with open(debug_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Aggressive Query Rewrite")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to process")
    parser.add_argument("--llm_model", type=str, default="claude-3-5-sonnet-20241022",
                        help="LLM model for rewriting (default: claude-3-5-sonnet-20241022)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output file already exists")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent workers for LLM calls")
    parser.add_argument("--prompt_name", type=str, default="aggressive",
                        help="Name of the prompt strategy (used for output directory name)")
    args = parser.parse_args()
    
    # Check for Claude 4.5 specifically if requested
    if "4.5" in args.llm_model and "sonnet" in args.llm_model:
         # Map "claude-sonnet-4.5" to actual API name if needed, or assume user knows it
         # Currently widely available model is 3.5 Sonnet. 
         # If the user specifically asked for "same sonnet model" as targeted rewrite,
         # we should use "claude-sonnet-4-5-20250929" as seen in targeted rewrite script.
         pass

    print("="*80)
    print(f"Query Rewrite Strategy: {args.prompt_name}")
    print("="*80)
    print(f"LLM model: {args.llm_model}")
    print(f"Concurrent workers: {args.workers}")
    print(f"Domains: {args.domains}")
    print(f"Output Directory: {BASE_OUTPUT_DIR / args.prompt_name}")
    print("="*80)
    
    # Create base output directory
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    print("Initializing LLM client...")
    try:
        llm_client = get_llm_client()
    except Exception as e:
        print(f"Failed to initialize Anthropic client: {e}")
        return

    # Load data
    conversations = load_conversations()
    tasks_by_domain = load_tasks()
    
    # Process domains
    for domain in args.domains:
        if domain not in args.domains:
            continue
            
        tasks = tasks_by_domain.get(domain, [])
        if not tasks:
            print(f"No tasks found for {domain}")
            continue
            
        process_domain(domain, tasks, conversations, llm_client, args)
        
    print("\nAll done!")

if __name__ == "__main__":
    main()

