"""
Utility functions for baseline query rewrite experiments with Claude Sonnet.

This module uses FULL conversation history (no filtering) to establish
a fair baseline for comparison with the targeted rewrite approach.
"""

import json
import os
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

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

# Claude Sonnet configuration
SONNET_CONFIG = {
    "model_name": "claude-sonnet-4-20250514",
    "api_model_id": "claude-sonnet-4-20250514",
    "api_endpoint": "https://api.anthropic.com/v1/messages",
    "max_tokens": 2048,  # High limit to avoid truncation
    "temperature": 0.0,
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

def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found. Please set it in .env")
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
# LLM Calls (Rewrite and Generate) - Claude Sonnet with FULL context
# =============================================================================

def rewrite_query_sonnet(
    api_key: str,
    current_query: str,
    full_history: List[Dict[str, str]],
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """Rewrite the query using Claude Sonnet with FULL conversation history."""
    if not full_history:
        return current_query
    
    # Build history text from ALL turns (no filtering)
    history_lines = []
    for turn in full_history:
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
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": SONNET_CONFIG['api_model_id'],
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                SONNET_CONFIG['api_endpoint'],
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            rewritten = result['content'][0]['text'].strip()
            rewritten = rewritten.strip('"').strip("'")
            if rewritten.lower().startswith("rewritten"):
                rewritten = rewritten.split(":", 1)[-1].strip()
            
            return rewritten if rewritten else current_query
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"  Error rewriting query: {e}")
                return current_query
    
    return current_query


def generate_response_sonnet(
    api_key: str,
    current_query: str,
    contexts: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """Generate a response using Claude Sonnet from retrieved documents."""
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
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": SONNET_CONFIG['api_model_id'],
        "max_tokens": 2048,  # High limit to avoid truncation
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                SONNET_CONFIG['api_endpoint'],
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated = result['content'][0]['text'].strip()
            return generated if generated else "I do not have specific information."
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
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

