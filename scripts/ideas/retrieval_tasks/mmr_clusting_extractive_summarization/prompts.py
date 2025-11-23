#!/usr/bin/env python3
"""
Prompts for MMR Cluster-Based Query Rewriting
"""

from typing import List, Dict, Any


def create_rewrite_prompt(current_query: str, selected_sentences: List[Dict[str, Any]]) -> str:
    """
    Create prompt for LLM to rewrite query with context.
    
    Args:
        current_query: The current turn's query
        selected_sentences: Sentences selected via MMR (with 'sentence' field)
    
    Returns:
        Formatted prompt string
    """
    # Format context from selected sentences
    context_lines = [f"- {s['sentence']}" for s in selected_sentences]
    context = '\n'.join(context_lines)
    
    # Create prompt
    prompt = f"""Given the following conversation, please reword the final utterance from the user into a single utterance that does not need the history to understand the user's intent. Output in proper JSON format indicating the "class" (standalone or non-standalone) and the "reworded version" of the last utterance. Use this format: {{"class": "type of last utterance", "reworded version": "the last utterance rewritten into a standalone question, IF NEEDED"}}.

In your rewording of the last utterance, do not do any unnecessary rephrasing or introduction of new terms or concepts that were not mentioned in the prior part of the conversation. Be minimal, by staying as close as possible to the shape and meaning of the last user utterance. If the last user utterance is already clear and standalone, the reworded version should be THE SAME as the last user utterance, and the class should be 'standalone'.

Conversation history:
{context}

User: {current_query}

ASSISTANT:"""
    
    return prompt

