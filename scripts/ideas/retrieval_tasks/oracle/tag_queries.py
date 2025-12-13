#!/usr/bin/env python3
"""
Tag queries with Question Types using an LLM (Claude via Anthropic).

This script:
1. Iterates through all tasks in `cleaned_data`.
2. Sends the user query to an LLM to classify its 'Question Type'.
3. Saves the tagged data (preserving metadata) to `scripts/ideas/retrieval_tasks/oracle/tagged_queries/`.

Usage:
    python tag_queries.py
"""

import sys
import json
import os
import argparse
import time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
CLEANED_DATA_DIR = project_root / "cleaned_data" / "tasks"
CONVERSATIONS_DIR = project_root / "cleaned_data" / "conversations"
OUTPUT_DIR = script_dir / "tagged_queries"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Question Types Definition
QUESTION_TYPES = [
    "Comparative",
    "Composite",
    "Explanation",
    "Factoid",
    "How-To",
    "Keyword",
    "Non-Question",
    "Opinion",
    "Summarization",
    "Troubleshooting"
]

SYSTEM_PROMPT = f"""You are an expert query analyzer. Your task is to classify the user's query into 1 to 3 of the following categories, ordered by relevance:

{json.dumps(QUESTION_TYPES, indent=2)}

Definitions:
- Comparative: Asks for a comparison between two or more items (e.g., "difference between X and Y").
- Composite: A complex query with multiple distinct parts or steps.
- Explanation: Asks for an explanation of a concept or process (e.g., "Why...", "How does X work").
- Factoid: Asks for a specific fact or entity (e.g., "Who is...", "What is the capital of...").
- How-To: Asks for instructions on performing a task.
- Keyword: A sequence of keywords without grammatical structure (e.g., "python list sort").
- Non-Question: A statement or command that isn't strictly a question but implies a search intent.
- Opinion: Asks for subjective judgment or best practices without a single factual answer.
- Summarization: Asks for a summary of a topic or document.
- Troubleshooting: Asks for help resolving a specific error or problem.

Output ONLY a JSON list of strings (e.g. ["Factoid", "Keyword"]). Do not explain.
"""

def get_client():
    """Initialize Anthropic client."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found. Please set it in .env")
        sys.exit(1)
        
    return anthropic.Anthropic(api_key=api_key)

def classify_query(client, query, model):
    """Classify a single query."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Query: {query}\n\nCategory:"}
            ]
        )
        response_text = message.content[0].text.strip()
        
        # Clean up response (remove code blocks etc)
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        try:
            categories = json.loads(response_text)
            if isinstance(categories, str):
                categories = [categories] # Handle single string response
        except json.JSONDecodeError:
            # Fallback for non-JSON response (comma separated)
            categories = [c.strip() for c in response_text.split(',')]
            
        # Validate
        valid_categories = [c for c in categories if c in QUESTION_TYPES]
        
        if not valid_categories:
            return "Unknown"
            
        # Return list of valid categories (up to 3)
        return valid_categories[:3]
        
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print(f"Error classifying query: {e}")
        return "Error"

def get_history(conversation_id, current_turn_id):
    """Retrieve conversation history up to the current turn."""
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    history = []
    
    if not conv_file.exists():
        return history
        
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        messages = data.get('messages', [])
        
        current_turn_count = 0
        for msg in messages:
            if msg['speaker'] == 'user':
                current_turn_count += 1
                if current_turn_count == current_turn_id:
                    break # Stop before adding the current query
            
            history.append(f"{msg['speaker']}: {msg.get('text', '')}")
            
    except Exception as e:
        print(f"Error reading history for {conversation_id}: {e}")
        
    return history

def process_file(json_file, output_domain_dir, model):
    """Process a single file: read, classify, save."""
    # Check if output already exists
    output_file = output_domain_dir / json_file.name
    if output_file.exists():
        return # Skip

    # Initialize client per thread (Anthropic client is thread safe but creating new one is cheap)
    # Actually, let's create it once per thread or just pass it?
    # Anthropic client creates a session.
    # Let's create it inside to be safe with threads.
    client = get_client()

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract query (last turn user utterance)
        query = None
        if 'user' in data:
            if 'text' in data['user']:
                query = data['user']['text']
            elif 'utterance' in data['user']:
                query = data['user']['utterance']
        
        if not query and 'turns' in data: # Fallback for conversation format
            for turn in reversed(data['turns']):
                if turn['role'] == 'user':
                    query = turn['content']
                    break
        
        if not query:
            return
            
        # Get History
        history = []
        conversation_id = data.get('conversation_id')
        turn_id = data.get('turn_id')
        
        if conversation_id and turn_id:
            history = get_history(conversation_id, turn_id)
            
        # Construct Prompt with History
        history_text = "\n".join(history[-10:]) # Keep last 5 turns (10 messages) context
        if history_text:
            prompt = f"Conversation History:\n{history_text}\n\nCurrent Query: {query}"
        else:
            prompt = query
        
        # Classify
        category = classify_query(client, prompt, model)
        
        # Add tag to data
        if 'oracle_metadata' not in data:
            data['oracle_metadata'] = {}
        data['oracle_metadata']['predicted_question_type'] = category
        data['oracle_metadata']['tagging_model'] = model
        data['oracle_metadata']['history_used'] = bool(history)
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"  Error processing {json_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Tag queries with Question Types")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929", help="LLM Model ID")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Workers: {args.workers}")
    
    # Check for API Key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found. Please set it in .env")
        sys.exit(1)
    
    for domain in args.domains:
        print(f"\nProcessing domain: {domain}")
        input_dir = CLEANED_DATA_DIR / domain
        output_domain_dir = OUTPUT_DIR / domain
        output_domain_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"  Input directory not found: {input_dir}")
            continue
            
        json_files = list(input_dir.glob("*.json"))
        print(f"  Found {len(json_files)} tasks.")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Create a list of futures
            futures = [
                executor.submit(process_file, json_file, output_domain_dir, args.model)
                for json_file in json_files
            ]
            
            # Use tqdm to show progress
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Tagging {domain}"):
                pass

    print("\nDone!")

if __name__ == "__main__":
    main()
