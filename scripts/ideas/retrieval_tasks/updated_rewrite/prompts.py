"""
Prompts for query rewriting strategies.
"""

def get_rewrite_prompt(prompt_name: str, history_text: str, current_query: str) -> dict:
    """
    Get prompt messages based on strategy name.
    """
    if prompt_name == "aggressive":
        return get_aggressive_rewrite_prompt(history_text, current_query)
    elif prompt_name == "informative":
        return get_informative_rewrite_prompt(history_text, current_query)
    elif prompt_name == "rewrite_then_edit":
        # Note: For this strategy, 'current_query' argument is expected to be the BASELINE REWRITE
        return get_rewrite_then_edit_prompt(history_text, current_query)
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

def get_informative_rewrite_prompt(history_text: str, current_query: str) -> dict:
    """
    Get the system and user messages for the informative rewrite strategy.
    
    Args:
        history_text: Formatted conversation history
        current_query: Current user query
        
    Returns:
        Dictionary with 'system' and 'user' prompt strings
    """
    system_prompt = """You are an expert query rewriter for information retrieval systems.

Your task is to rewrite conversational queries to be:
1. CORRECT: Preserve original meaning
2. CLEAR: Standalone, resolve all pronouns/references
3. INFORMATIVE: Add valuable context to help retrieval
4. NON-REDUNDANT: Focus only on current query intent

=== INSTRUCTIONS ===
Rewrite the current query following these steps:

Step 1 - Resolve Context:
- Replace pronouns with actual entities
- Resolve "this", "that", "it" references
- Make query grammatically complete

Step 2 - Add Informative Context:
- Add key entities and their attributes
- Add relevant background information mentioned in conversation
- Add synonyms and related terms that might appear in documents
- Add domain-specific vocabulary

Step 3 - Optimize for Retrieval:
- Think about what terms would appear in relevant documents
- Add both formal and informal variants of key terms
- Include related concepts not explicitly mentioned but implied

=== EXAMPLES ===

Example 1:
Conversation: ["Who is Marie Curie?"]
Current: "What did she discover?"
Rewrite: "What did Marie Curie discover? Marie Curie physicist chemist Nobel Prize winner Poland France radioactivity scientific discoveries achievements polonium radium research contributions"

Example 2:
Conversation: ["What caused the 2008 financial crisis?"]
Current: "What happened after that?"
Rewrite: "What happened after 2008 financial crisis? Financial crisis 2008 2009 2010 aftermath recovery economic impact consequences recession Great Recession banking reforms policy changes"

Example 3:
Conversation: ["Tell me about Python"]
Current: "How does it compare to Java?"
Rewrite: "How does Python compare to Java? Python Java programming languages comparison differences similarities advantages disadvantages syntax performance use cases development speed type system"

=== OUTPUT ===
Provide ONLY the rewritten query, no explanation."""

    user_message = f"""=== CONVERSATION HISTORY ===
{history_text}

=== CURRENT QUERY ===
{current_query}

=== OUTPUT ===
Provide ONLY the rewritten query, no explanation:"""

    return {
        "system": system_prompt,
        "user": user_message
    }

def get_rewrite_then_edit_prompt(history_text: str, baseline_rewrite: str) -> dict:
    """
    Get prompt to edit/expand an existing rewrite.
    """
    system_prompt = """You are a Search Engine Query Optimizer.
    
Your task is to EDIT an existing search query to make it MORE informative and effective for retrieval.
The goal is to expand the query with relevant context, synonyms, and related entities without changing the core intent.

Instructions:
1. START with the provided 'Baseline Query'.
2. EXPAND it by adding:
   - Related entities and attributes mentioned in the conversation history.
   - Background context that helps retrieval (even if implied).
   - Synonyms and related terms that might appear in target documents.
3. FORMAT the output as a keyword-rich search string.

Example:
Baseline: "What did Marie Curie discover?"
Context: (Conversation about her life)
Output: "What did Marie Curie discover? Marie Curie physicist chemist Nobel Prize radioactivity polonium radium scientific discoveries achievements contributions research"

Output ONLY the final expanded query."""

    user_message = f"""Conversation History:
{history_text}

Baseline Query: {baseline_rewrite}

Output ONLY the expanded query:"""

    return {
        "system": system_prompt,
        "user": user_message
    }

