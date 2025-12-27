"""
Utility functions for targeted query rewrite experiments.

This module contains shared functionality for:
- Embedding and similarity computation
- LLM API calls (Mixtral via Together AI)
- Elasticsearch retrieval (ELSER)
- Conversation and task loading
- Context filtering
"""

import json
import os
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Paths and Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
CONVERSATIONS_DIR = PROJECT_ROOT / "cleaned_data" / "conversations"
TASKS_DIR = PROJECT_ROOT / "cleaned_data" / "tasks"
OUTPUT_DIR = SCRIPT_DIR / "retrieval_results"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# Mixtral 8x7B configuration (same as paper baseline for fair comparison)
MIXTRAL_CONFIG = {
    "model_name": "mixtral_8x7b_instruct",
    "api_model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "api_endpoint": "https://api.together.xyz/v1/chat/completions",
    "max_tokens": 2048,  # High limit to avoid truncation
    "temperature": 0.0,
    "top_p": 1.0,
}

# ELSER index mappings
ELSER_INDEX_MAPPING = {
    'clapnq': "mtrag-clapnq-elser-512-100-reindexed",
    'cloud': "mtrag-cloud-elser-512-100",
    'fiqa': "mtrag-fiqa-elser-512-100",
    'govt': "mtrag-govt-elser-512-100",
}

COLLECTION_MAPPING = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'govt': 'mt-rag-govt-elser-512-100-20240611',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
}


# =============================================================================
# Model Loading
# =============================================================================

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


def get_together_api_key() -> str:
    """Get Together AI API key from environment."""
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        print("Error: TOGETHER_API_KEY not found. Please set it in .env")
        sys.exit(1)
    return api_key


def get_elasticsearch_client():
    """Get Elasticsearch client."""
    from elasticsearch import Elasticsearch
    
    es_url = os.getenv('ES_URL')
    api_key = os.getenv('ES_API_KEY')
    
    if not es_url or not api_key:
        print("Error: ES_URL and ES_API_KEY must be set in .env")
        sys.exit(1)
    
    es = Elasticsearch(es_url, api_key=api_key, request_timeout=60)
    
    if not es.ping():
        print("Error: Failed to connect to Elasticsearch")
        sys.exit(1)
    
    return es


# =============================================================================
# Similarity and Context Filtering
# =============================================================================

def compute_similarity(query_emb: np.ndarray, turn_emb: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    turn_norm = turn_emb / (np.linalg.norm(turn_emb) + 1e-9)
    return float(np.dot(query_norm, turn_norm))


def group_history_into_turns(conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Group conversation history into turn pairs (user + agent response)."""
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
            turn["agent"] = conversation_history[i].get("text", "")
            i += 1
        
        if turn["user"] or turn["agent"]:
            turn_pairs.append(turn)
    
    return turn_pairs


def select_relevant_turns(
    current_query: str,
    conversation_history: List[Dict[str, str]],
    embedding_model,
    similarity_threshold: float = 0.3,
    include_last_turn: bool = True
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Select relevant turns from conversation history based on semantic similarity."""
    if not conversation_history:
        return [], {"num_history_turns": 0, "selected_turns": 0}
    
    turn_pairs = group_history_into_turns(conversation_history)
    
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
        similarities.append({"turn_idx": idx, "similarity": sim})
    
    # Select turns above threshold
    selected_indices = set()
    
    if include_last_turn and turn_pairs:
        selected_indices.add(len(turn_pairs) - 1)
    
    for sim_info in similarities:
        if sim_info["similarity"] >= similarity_threshold:
            selected_indices.add(sim_info["turn_idx"])
    
    selected_indices = sorted(selected_indices)
    
    # Build selected turns list
    selected_turns = []
    for idx in selected_indices:
        tp = turn_pairs[idx]
        if tp["user"]:
            selected_turns.append({"speaker": "user", "text": tp["user"]})
        if tp["agent"]:
            selected_turns.append({"speaker": "agent", "text": tp["agent"]})
    
    analysis = {
        "num_history_turns": len(turn_pairs),
        "selected_turns": len(selected_indices),
        "similarities": similarities
    }
    
    return selected_turns, analysis


# =============================================================================
# LLM Calls (Rewrite and Generate)
# =============================================================================

def rewrite_query_mixtral(
    api_key: str,
    current_query: str,
    relevant_history: List[Dict[str, str]],
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """Rewrite the query using Mixtral 8x7B via Together AI."""
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
        "model": MIXTRAL_CONFIG['api_model_id'],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                MIXTRAL_CONFIG['api_endpoint'],
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            rewritten = result['choices'][0]['message']['content'].strip()
            rewritten = rewritten.strip('"').strip("'")
            if rewritten.lower().startswith("rewritten"):
                rewritten = rewritten.split(":", 1)[-1].strip()
            
            return rewritten if rewritten else current_query
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"  Error rewriting query: {e}")
                return current_query
    
    return current_query


def generate_response_mixtral(
    api_key: str,
    current_query: str,
    contexts: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """Generate a response using Mixtral from retrieved documents."""
    # Build prompt following paper format (Section D.2)
    prompt = """Given one or more documents and a user query, generate a response to the query using less than 150 words that is grounded in the provided documents. If no answer can be found in the documents, say, "I do not have specific information"

"""
    
    # Add passages
    for i, ctx in enumerate(contexts[:5], 1):  # Top 5 passages
        prompt += f"PASSAGE {i}\n"
        prompt += ctx.get('text', '') + "\n\n"
    
    # Add conversation history
    for turn in conversation_history:
        speaker = turn.get("speaker", "user").capitalize()
        text = turn.get("text", "")
        prompt += f"{speaker}: {text}\n"
    
    # Add current query
    prompt += f"User: {current_query}\n"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MIXTRAL_CONFIG['api_model_id'],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,  # High limit to avoid truncation
        "temperature": 0.0,
        "top_p": 1.0,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                MIXTRAL_CONFIG['api_endpoint'],
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated = result['choices'][0]['message']['content'].strip()
            return generated if generated else "I do not have specific information."
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"  Error generating response: {e}")
                return "I do not have specific information."
    
    return "I do not have specific information."


# =============================================================================
# Retrieval (ELSER)
# =============================================================================

def retrieve_documents_elser(
    es,
    index_name: str,
    query_text: str,
    top_k: int = 10,
    delay: float = 0.3
) -> List[Dict[str, Any]]:
    """Retrieve documents using ELSER."""
    # Clean query text
    clean_text = query_text.replace('|user|: ', '').replace('|user|:', '').strip()
    
    query_body = {
        "query": {
            "text_expansion": {
                "ml.tokens": {
                    "model_id": ".elser-2-elastic",
                    "model_text": clean_text
                }
            }
        },
        "size": top_k,
        "_source": ["text", "title", "url"]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = es.search(index=index_name, body=query_body)
            
            contexts = []
            for hit in response['hits']['hits']:
                contexts.append({
                    'document_id': hit['_id'],
                    'score': float(hit['_score']),
                    'text': hit['_source'].get('text', ''),
                    'title': hit['_source'].get('title', ''),
                    'source': hit['_source'].get('url', '')
                })
            
            time.sleep(delay)  # Rate limiting
            return contexts
            
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"  Retrieval error: {e}")
                return []
    
    return []


# =============================================================================
# Data Loading
# =============================================================================

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


def load_tasks_by_conversation(domain: str) -> Dict[str, List[Dict]]:
    """Load all tasks for a domain, grouped by conversation_id and sorted by turn_id."""
    tasks_dir = TASKS_DIR / domain
    if not tasks_dir.exists():
        return {}
    
    tasks_by_conv = defaultdict(list)
    
    for task_file in tasks_dir.glob("*.json"):
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            conv_id = str(task_data.get("conversation_id"))
            tasks_by_conv[conv_id].append(task_data)
        except Exception as e:
            print(f"Error loading {task_file}: {e}")
    
    # Sort by turn_id
    for conv_id in tasks_by_conv:
        tasks_by_conv[conv_id].sort(key=lambda x: x.get("turn_id", 0))
    
    return dict(tasks_by_conv)

