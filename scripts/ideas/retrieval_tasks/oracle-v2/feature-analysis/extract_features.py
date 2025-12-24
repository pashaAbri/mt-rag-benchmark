#!/usr/bin/env python3
"""
Extract features from task data for routing analysis.

Extracts query characteristics, enrichments, and conversation context features
from the cleaned task data.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re


def extract_query_features(query_text: str) -> Dict[str, any]:
    """Extract features from query text."""
    if not query_text:
        return {
            'query_length_chars': 0,
            'query_length_words': 0,
            'has_question_mark': False,
            'has_wh_word': False,
            'num_capitalized_words': 0,
            'num_numbers': 0,
        }
    
    words = query_text.split()
    
    # Check for question mark
    has_question_mark = '?' in query_text
    
    # Check for wh-words (what, who, where, when, why, how, which)
    wh_pattern = r'\b(what|who|where|when|why|how|which|whose|whom)\b'
    has_wh_word = bool(re.search(wh_pattern, query_text.lower()))
    
    # Count capitalized words (excluding first word)
    num_capitalized = sum(1 for w in words[1:] if w and w[0].isupper())
    
    # Count numbers
    num_numbers = len(re.findall(r'\d+', query_text))
    
    return {
        'query_length_chars': len(query_text),
        'query_length_words': len(words),
        'has_question_mark': has_question_mark,
        'has_wh_word': has_wh_word,
        'num_capitalized_words': num_capitalized,
        'num_numbers': num_numbers,
    }


def extract_enrichment_features(enrichments: Dict) -> Dict[str, any]:
    """Extract features from enrichments."""
    features = {
        'answerability': None,
        'question_type': None,
        'multi_turn_type': None,
    }
    
    if enrichments:
        # Answerability (take first if list)
        answerability = enrichments.get('Answerability', [])
        if answerability and isinstance(answerability, list):
            features['answerability'] = answerability[0] if answerability else None
        elif answerability:
            features['answerability'] = answerability
        
        # Question Type (take first if list)
        question_type = enrichments.get('Question Type', [])
        if question_type and isinstance(question_type, list):
            features['question_type'] = question_type[0] if question_type else None
        elif question_type:
            features['question_type'] = question_type
        
        # Multi-Turn Type (take first if list)
        multi_turn = enrichments.get('Multi-Turn', [])
        if multi_turn and isinstance(multi_turn, list):
            features['multi_turn_type'] = multi_turn[0] if multi_turn else None
        elif multi_turn:
            features['multi_turn_type'] = multi_turn
    
    return features


def load_task_data(task_file: Path) -> Optional[Dict]:
    """Load a single task file."""
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_conversation_data(conversations_file: Path) -> Dict[str, Dict]:
    """Load conversation data to get history."""
    conversations = {}
    try:
        with open(conversations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for conv in data:
                    conv_id = str(conv.get('conversation_id', ''))
                    if conv_id:
                        conversations[conv_id] = conv
            elif isinstance(data, dict):
                conversations = data
    except (json.JSONDecodeError, IOError):
        pass
    return conversations


def extract_conversation_features(
    task_data: Dict,
    conversations: Dict[str, Dict]
) -> Dict[str, any]:
    """Extract conversation history features."""
    features = {
        'conversation_length': 0,
        'num_previous_turns': 0,
        'previous_turn_query_length': None,
        'previous_turn_question_type': None,
        'previous_turn_multi_turn_type': None,
    }
    
    conversation_id = str(task_data.get('conversation_id', ''))
    turn_id = task_data.get('turn_id', 1)
    
    if conversation_id in conversations:
        conv = conversations[conversation_id]
        messages = conv.get('messages', [])
        
        # Count total conversation length
        user_messages = [m for m in messages if m.get('speaker') == 'user']
        features['conversation_length'] = len(user_messages)
        features['num_previous_turns'] = turn_id - 1
        
        # Get previous turn characteristics
        if turn_id > 1 and len(user_messages) >= turn_id - 1:
            prev_message = user_messages[turn_id - 2]  # 0-indexed
            prev_text = prev_message.get('text', '')
            prev_enrichments = prev_message.get('enrichments', {})
            
            features['previous_turn_query_length'] = len(prev_text.split())
            
            # Previous turn enrichments
            prev_question_type = prev_enrichments.get('Question Type', [])
            if prev_question_type and isinstance(prev_question_type, list):
                features['previous_turn_question_type'] = prev_question_type[0] if prev_question_type else None
            elif prev_question_type:
                features['previous_turn_question_type'] = prev_question_type
            
            prev_multi_turn = prev_enrichments.get('Multi-Turn', [])
            if prev_multi_turn and isinstance(prev_multi_turn, list):
                features['previous_turn_multi_turn_type'] = prev_multi_turn[0] if prev_multi_turn else None
            elif prev_multi_turn:
                features['previous_turn_multi_turn_type'] = prev_multi_turn
    
    return features


def extract_task_features(task_data: Dict, conversations: Dict[str, Dict] = None) -> Dict[str, any]:
    """Extract all features from a task."""
    features = {
        'task_id': task_data.get('task_id'),
        'conversation_id': task_data.get('conversation_id'),
        'turn_id': task_data.get('turn_id'),
        'domain': task_data.get('domain'),
    }
    
    # Extract user query features
    user_data = task_data.get('user', {})
    query_text = user_data.get('text', '')
    query_features = extract_query_features(query_text)
    features.update(query_features)
    
    # Extract enrichments
    enrichments = user_data.get('enrichments', {})
    enrichment_features = extract_enrichment_features(enrichments)
    features.update(enrichment_features)
    
    # Conversation context features
    features['is_first_turn'] = (task_data.get('turn_id', 0) == 1)
    
    # Extract conversation history features if available
    if conversations:
        conv_features = extract_conversation_features(task_data, conversations)
        features.update(conv_features)
    
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from task data"
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="cleaned_data/tasks",
        help="Directory containing task JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="task_features.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=['all', 'clapnq', 'cloud', 'fiqa', 'govt'],
        default='all',
        help="Domain to process"
    )
    parser.add_argument(
        "--conversations-file",
        type=str,
        default="human/conversations/conversations.json",
        help="Path to conversations JSON file"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[4]
    tasks_dir = (project_root / args.tasks_dir).resolve()
    conversations_file = project_root / args.conversations_file
    output_file = script_dir / args.output
    
    print("=" * 80)
    print("EXTRACTING TASK FEATURES")
    print("=" * 80)
    print(f"Tasks directory: {tasks_dir}")
    print(f"Conversations file: {conversations_file}")
    print(f"Domain: {args.domain}")
    print(f"Output: {output_file}")
    
    # Load conversation data for history features
    print("\nLoading conversation data...")
    conversations = {}
    if conversations_file.exists():
        conversations = load_conversation_data(conversations_file)
        print(f"  Loaded {len(conversations)} conversations")
    else:
        print("  Warning: Conversations file not found, skipping history features")
    
    # Collect task files
    task_files = []
    if args.domain == 'all':
        for domain_dir in tasks_dir.iterdir():
            if domain_dir.is_dir():
                task_files.extend(domain_dir.glob("*.json"))
    else:
        domain_dir = tasks_dir / args.domain
        if domain_dir.exists():
            task_files.extend(domain_dir.glob("*.json"))
    
    print(f"\nFound {len(task_files)} task files")
    
    # Extract features
    print("Extracting features...")
    all_features = {}
    failed = 0
    
    for task_file in task_files:
        task_data = load_task_data(task_file)
        if not task_data:
            failed += 1
            continue
        
        features = extract_task_features(task_data, conversations)
        task_id = features.get('task_id')
        
        if task_id:
            all_features[task_id] = features
    
    print(f"Extracted features for {len(all_features)} tasks")
    if failed > 0:
        print(f"Failed to process {failed} files")
    
    # Save results
    output_data = {
        'domain': args.domain,
        'total_tasks': len(all_features),
        'features': all_features
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nFeatures saved to: {output_file}")


if __name__ == "__main__":
    main()

