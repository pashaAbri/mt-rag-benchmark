#!/usr/bin/env python3
"""
Context Builder for Multi-Turn RAG - Incremental Approach

This module supports building context incrementally as we generate responses.
We don't use pre-existing conversation data - instead, we build our own context
as we go through the experiment turn by turn.

Workflow:
1. Load lastturn queries and organize by conversation/turn
2. Process Turn 1: No context (no rewriting), generate responses, save them
3. Process Turn 2: Load Turn 1 responses as context, rewrite queries
4. Process Turn 3: Load Turn 1+2 responses as context, rewrite queries
... and so on
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_queries_from_cleaned_data(domain: str) -> Dict[str, str]:
    """
    Load queries from cleaned_data/tasks/{domain}/ directory.
    
    Args:
        domain: Domain name (e.g., 'clapnq')
    
    Returns:
        Dict mapping task_id to query text
    """
    queries = {}
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    tasks_dir = repo_root / 'cleaned_data' / 'tasks' / domain
    
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
        
    print(f"Scanning {tasks_dir}...")
    for task_file in tasks_dir.glob('*.json'):
        try:
            with open(task_file, 'r') as f:
                data = json.loads(f.read())
                task_id = data['task_id']
                # The query is in user.text
                query = data['user']['text']
                queries[task_id] = query
        except Exception as e:
            print(f"Error reading {task_file}: {e}")
            
    return queries


def organize_by_conversation(queries: Dict[str, str]) -> Dict[str, List[Dict]]:
    """
    Organize queries by conversation ID and turn number.
    
    Task ID format: {conversation_id}<::>{turn_number}
    
    Args:
        queries: Dict mapping task_id to query text
    
    Returns:
        Dict mapping conversation_id to list of turn dicts sorted by turn:
        {
            'task_id': str,
            'turn': int,
            'query': str
        }
    """
    conversations = defaultdict(list)
    
    for task_id, query_text in queries.items():
        # Parse task_id: format is conversation_id<::>turn
        if '<::>' not in task_id:
            print(f"Warning: Invalid task_id format: {task_id}")
            continue
        
        parts = task_id.split('<::>')
        conv_id = parts[0]
        turn = int(parts[1])
        
        conversations[conv_id].append({
            'task_id': task_id,
            'turn': turn,
            'query': query_text
        })
    
    # Sort each conversation by turn number
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['turn'])
    
    return dict(conversations)


def get_turn_tasks(
    conversations: Dict[str, List[Dict]], 
    turn_number: int
) -> List[Dict]:
    """
    Get all tasks for a specific turn number across all conversations.
    
    Args:
        conversations: Output from organize_by_conversation()
        turn_number: Turn number to extract (1-indexed)
    
    Returns:
        List of task dicts for the specified turn:
        {
            'task_id': str,
            'conv_id': str,
            'turn': int,
            'query': str
        }
    """
    tasks = []
    
    for conv_id, turns in conversations.items():
        for turn_data in turns:
            if turn_data['turn'] == turn_number:
                tasks.append({
                    'task_id': turn_data['task_id'],
                    'conv_id': conv_id,
                    'turn': turn_data['turn'],
                    'query': turn_data['query']
                })
    
    return tasks


def get_max_turns(conversations: Dict[str, List[Dict]]) -> int:
    """
    Get the maximum turn number across all conversations.
    
    Args:
        conversations: Output from organize_by_conversation()
    
    Returns:
        Maximum turn number
    """
    max_turn = 0
    for turns in conversations.values():
        for turn_data in turns:
            max_turn = max(max_turn, turn_data['turn'])
    return max_turn


def build_history_for_task(
    task_id: str,
    conversations: Dict[str, List[Dict]],
    agent_responses: Dict[str, str]
) -> List[Dict]:
    """
    Build conversation history for a specific task.
    
    History includes all previous Q+A pairs from the same conversation.
    
    Args:
        task_id: Task ID in format {conv_id}<::>{turn}
        conversations: Output from organize_by_conversation()
        agent_responses: Dict mapping task_id to agent response
    
    Returns:
        List of history items in chronological order:
        [
            {'speaker': 'user', 'text': 'Q1'},
            {'speaker': 'agent', 'text': 'A1'},
            {'speaker': 'user', 'text': 'Q2'},
            {'speaker': 'agent', 'text': 'A2'},
            ...
        ]
    """
    # Parse task_id
    if '<::>' not in task_id:
        return []
    
    conv_id, turn_str = task_id.split('<::>')
    current_turn = int(turn_str)
    
    # Get conversation turns
    if conv_id not in conversations:
        return []
    
    conversation = conversations[conv_id]
    
    # Build history from previous turns
    history = []
    for turn_data in conversation:
        if turn_data['turn'] >= current_turn:
            break
        
        # Add user question
        history.append({
            'speaker': 'user',
            'text': turn_data['query']
        })
        
        # Add agent response if available
        prev_task_id = turn_data['task_id']
        if prev_task_id in agent_responses:
            history.append({
                'speaker': 'agent',
                'text': agent_responses[prev_task_id]
            })
        else:
            # If we don't have a response (e.g. skipped or error), we can't add it.
            # This simulates "no history" for that turn or partial history.
            pass
    
    return history


def get_dataset_statistics(
    conversations: Dict[str, List[Dict]]
) -> Dict:
    """
    Get statistics about the dataset.
    
    Args:
        conversations: Output from organize_by_conversation()
    
    Returns:
        Dict with statistics
    """
    total_tasks = sum(len(turns) for turns in conversations.values())
    max_turn = get_max_turns(conversations)
    
    # Count tasks per turn
    tasks_per_turn = defaultdict(int)
    for turns in conversations.values():
        for turn_data in turns:
            tasks_per_turn[turn_data['turn']] += 1
    
    return {
        'num_conversations': len(conversations),
        'total_tasks': total_tasks,
        'max_turn': max_turn,
        'tasks_per_turn': dict(sorted(tasks_per_turn.items()))
    }


# ============================================================================
# Main workflow functions
# ============================================================================

def initialize_dataset(domain: str) -> Tuple[Dict[str, List[Dict]], Dict]:
    """
    Initialize dataset from cleaned_data tasks.
    
    This is the entry point for the incremental workflow.
    
    Args:
        domain: Domain name (e.g., 'clapnq')
    
    Returns:
        Tuple of (conversations_dict, statistics)
    """
    print(f"Loading queries for {domain} from cleaned_data...")
    queries = load_queries_from_cleaned_data(domain)
    print(f"  → Loaded {len(queries)} queries")
    
    print("Organizing by conversation...")
    conversations = organize_by_conversation(queries)
    
    stats = get_dataset_statistics(conversations)
    print(f"  → {stats['num_conversations']} conversations")
    print(f"  → {stats['total_tasks']} total tasks")
    print(f"  → {stats['max_turn']} maximum turns")
    print(f"  → Tasks per turn: {stats['tasks_per_turn']}")
    
    return conversations, stats


def prepare_turn_batch(
    conversations: Dict[str, List[Dict]],
    turn_number: int,
    agent_responses: Dict[str, str] = None
) -> List[Dict]:
    """
    Prepare a batch of tasks for a specific turn with their history.
    
    Args:
        conversations: Output from initialize_dataset()
        turn_number: Turn number to process
        agent_responses: Dict mapping previous task_id to generated agent response
    
    Returns:
        List of task dicts ready for processing:
        {
            'task_id': str,
            'conv_id': str,
            'turn': int,
            'query': str,
            'history': List[Dict]  # Empty for turn 1
        }
    """
    if agent_responses is None:
        agent_responses = {}

    # Get tasks for this turn
    tasks = get_turn_tasks(conversations, turn_number)
    
    if turn_number == 1:
        # No history for first turn
        for task in tasks:
            task['history'] = []
        return tasks
    
    # Build history for each task
    for task in tasks:
        task['history'] = build_history_for_task(
            task['task_id'],
            conversations,
            agent_responses
        )
    
    return tasks


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == '__main__':
    """Test the context builder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test context builder')
    parser.add_argument('--domain', type=str, required=True, 
                       choices=['clapnq', 'fiqa', 'govt', 'cloud'])
    args = parser.parse_args()
    
    # Initialize
    conversations, stats = initialize_dataset(args.domain)
    
    print(f"\n{'='*60}")
    print("Sample Turn Processing")
    print(f"{'='*60}\n")
    
    # Show Turn 1 example
    print("Turn 1 Tasks (no history):")
    turn1_tasks = prepare_turn_batch(conversations, 1)
    if turn1_tasks:
        sample = turn1_tasks[0]
        print(f"  Task ID: {sample['task_id']}")
        print(f"  Query: {sample['query'][:60]}...")
        print(f"  History: {len(sample['history'])} items (should be 0)")
    
    print()
    
    # Show Turn 2 example (without actual responses)
    if stats['max_turn'] >= 2:
        print("Turn 2 Tasks (would need Turn 1 responses for history):")
        turn2_tasks = prepare_turn_batch(conversations, 2)
        if turn2_tasks:
            sample = turn2_tasks[0]
            print(f"  Task ID: {sample['task_id']}")
            print(f"  Query: {sample['query'][:60]}...")
            print(f"  History: {len(sample['history'])} items (0 without response file)")
            print(f"  Note: Load agent responses to build history")
    
    print(f"\n{'='*60}")
    print("Context Builder Test Complete!")
    print(f"{'='*60}\n")
