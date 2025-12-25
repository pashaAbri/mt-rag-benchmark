"""
Prompts for query rewriting strategies.
"""

def get_rewrite_prompt(prompt_name: str, history_text: str, current_query: str) -> dict:
    """
    Get prompt messages based on strategy name.
    """
    if prompt_name == "aggressive":
        return get_aggressive_rewrite_prompt(history_text, current_query)
    else:
        # Default or fallback
        return get_aggressive_rewrite_prompt(history_text, current_query)

def get_aggressive_rewrite_prompt(history_text: str, current_query: str) -> dict:
    """
    Get the system and user messages for the aggressive rewrite strategy.
    
    Args:
        history_text: Formatted conversation history
        current_query: Current user query
        
    Returns:
        Dictionary with 'system' and 'user' prompt strings
    """
    system_prompt = """You are a Search Engine Query Optimizer. Your task is to rewrite the last user utterance into a fully standalone search query that will retrieve relevant documents.

Rules:
1. **RESOLVE ALL PRONOUNS**: Replace every pronoun (it, he, she, they, this, that) with the specific entity name it refers to from the conversation history.
2. **INJECT MISSING CONTEXT**: If the query implies a topic discussed earlier (e.g., "what about security?"), explicitly add the topic (e.g., "IBM Cloud security features").
3. **SPECIFY GENERIC TERMS**: Replace generic words like "the series", "the act", "the company" with their full proper names (e.g., "The Office US", "The Affordable Care Act").
4. **IGNORE NATURALNESS**: The output does not need to sound like a natural conversation. It must be an effective keyword-rich search query.
5. **NEVER OUTPUT THE SAME QUERY**: If the user's query relies on *any* previous context, you MUST modify it.

Output ONLY the rewritten query text, nothing else."""

    user_message = f"""Input Conversation:
{history_text}
User: {current_query}

Output ONLY the rewritten query text."""

    return {
        "system": system_prompt,
        "user": user_message
    }

