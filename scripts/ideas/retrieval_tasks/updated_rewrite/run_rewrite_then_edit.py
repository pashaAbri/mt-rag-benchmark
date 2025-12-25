#!/usr/bin/env python3
"""
Rewrite-Then-Edit Strategy

This script implements a two-step query strategy:
1. Start with the Baseline Rewrite (human/expert rewrite).
2. Edit/Expand it using an LLM to add context, synonyms, and related entities.

It processes all conversations and generates expanded queries using Claude Sonnet 4.5.
"""

import json
import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
import anthropic

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import prompts
from prompts import get_rewrite_prompt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

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
    """Load all conversations."""
    conversations = {}
    if not data_dir.exists():
        data_dir = Path("cleaned_data/conversations")
    
    print(f"Loading conversations from {data_dir}...")
    for f in tqdm(list(data_dir.glob("*.json")), desc="Loading conversations"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    for conv in data:
                        cid = str(conv.get('conversation_id', ''))
                        if cid: conversations[cid] = conv
                elif isinstance(data, dict):
                    cid = str(data.get('conversation_id', ''))
                    if not cid: cid = f.stem
                    if cid: conversations[cid] = data
        except Exception:
            pass
    return conversations

def load_baseline_rewrites(domain: str) -> Dict[str, str]:
    """Load baseline rewrites from human/retrieval_tasks."""
    rewrites = {}
    path = Path(f"human/retrieval_tasks/{domain}/{domain}_rewrite.jsonl")
    if not path.exists():
        # Try relative to project root
        path = Path(f"human/retrieval_tasks/{domain}/{domain}_rewrite.jsonl")
        
    if path.exists():
        print(f"Loading baseline rewrites for {domain} from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Remove |user|: prefix if present
                text = data['text'].replace('|user|: ', '').replace('|user|:', '').strip()
                rewrites[data['_id']] = text
    else:
        print(f"Warning: Baseline rewrites not found for {domain} at {path}")
        
    return rewrites

def load_tasks(tasks_dir: Path = Path("cleaned_data/tasks")) -> Dict[str, List[Dict]]:
    """Load tasks grouped by domain."""
    tasks_by_domain = {d: [] for d in DOMAINS}
    if not tasks_dir.exists(): tasks_dir = Path("cleaned_data/tasks")
    
    print(f"Loading tasks from {tasks_dir}...")
    for domain in DOMAINS:
        domain_dir = tasks_dir / domain
        if domain_dir.exists():
            for f in domain_dir.glob("*.json"):
                try:
                    with open(f, 'r') as file:
                        tasks_by_domain[domain].append(json.load(file))
                except:
                    pass
    return tasks_by_domain

def expand_query(client, baseline_query: str, history: List[Dict], model: str) -> str:
    """Expand the baseline query using the LLM."""
    if not baseline_query:
        return ""
        
    # Build history text
    history_lines = []
    for turn in history:
        speaker = "User" if turn.get("speaker") == "user" else "Assistant"
        history_lines.append(f"{speaker}: {turn.get('text', '')}")
    history_text = "\n".join(history_lines)
    
    # Get prompt
    prompts = get_rewrite_prompt("rewrite_then_edit", history_text, baseline_query)
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.0,
            system=prompts["system"],
            messages=[{"role": "user", "content": prompts["user"]}]
        )
        expanded = message.content[0].text.strip()
        # Cleanup
        expanded = expanded.strip('"').strip("'")
        if expanded.lower().startswith("output:"):
            expanded = expanded.split(":", 1)[-1].strip()
        return expanded
    except Exception as e:
        print(f"Error expanding query: {e}")
        return baseline_query

def process_domain(domain, tasks, conversations, baseline_rewrites, client, args):
    output_dir = BASE_OUTPUT_DIR / args.prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{domain}_{args.prompt_name}_rewrite.jsonl"
    
    if args.skip_existing and output_file.exists():
        print(f"Skipping {domain} (exists)")
        return

    results = []
    
    # Define worker
    def process_task(task):
        task_id = task.get('task_id')
        baseline = baseline_rewrites.get(task_id)
        
        if not baseline:
            # Fallback to original query if baseline missing
            baseline = task.get('user', {}).get('text', '')
            
        # Get history (up to current turn)
        conv_id = str(task.get('conversation_id'))
        turn_id = task.get('turn_id', 1)
        history = []
        
        if conv_id in conversations:
            msgs = conversations[conv_id].get('messages', [])
            user_turns = 0
            for m in msgs:
                if m.get('speaker') == 'user':
                    user_turns += 1
                    if user_turns == turn_id: break
                history.append({"speaker": m.get("speaker"), "text": m.get("text")})
        
        expanded = expand_query(client, baseline, history, args.llm_model)
        
        return {
            "_id": task_id,
            "text": expanded,
            "original": task.get('user', {}).get('text', ''),
            "baseline": baseline
        }

    print(f"Processing {len(tasks)} tasks for {domain}...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_task, tasks), total=len(tasks)))
        
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps({"_id": res["_id"], "text": f"|user|: {res['text']}"}) + "\n")
            
    # Debug file
    with open(output_dir / f"{domain}_{args.prompt_name}_rewrite_debug.jsonl", 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--llm_model", default="claude-3-5-sonnet-20241022")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prompt_name", default="rewrite_then_edit")
    args = parser.parse_args()
    
    client = get_llm_client()
    conversations = load_conversations()
    tasks_by_domain = load_tasks()
    
    for domain in args.domains:
        if domain not in tasks_by_domain: continue
        baseline_rewrites = load_baseline_rewrites(domain)
        process_domain(domain, tasks_by_domain[domain], conversations, baseline_rewrites, client, args)

if __name__ == "__main__":
    main()

