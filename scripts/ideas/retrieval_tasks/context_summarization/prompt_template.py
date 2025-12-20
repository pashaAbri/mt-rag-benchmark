"""
Prompt templates for SELF-multi-RAG style conversation summarization.

Based on the paper:
"SELF-multi-RAG: A Multi-Turn Conversational Retrieval-Augmented Generation Framework"

The key innovation is creating a 40-50 word SUMMARY + reformulated QUESTION
instead of traditional query rewriting (single question) or full conversation.

This approach:
1. Preserves important context from the conversation
2. Removes noise/irrelevant details  
3. Creates a self-contained retrieval query
4. Achieves +13.5% retrieval improvement over query rewriting (per paper)
"""

# Few-shot examples based on the paper's Table 13 and Figure 2
# Adapted for the MT-RAG benchmark domains

FEW_SHOT_EXAMPLES = [
    # Example 1: Wikipedia/ClapNQ domain (from paper)
    {
        "conversation": [
            {"speaker": "user", "text": "What was the first job John Sherman Cooper held?"},
            {"speaker": "agent", "text": "He was admitted to the bar by examination in 1928 and began practicing law in Somerset."},
            {"speaker": "user", "text": "What was the first office John Sherman Cooper ran for?"},
            {"speaker": "agent", "text": "After being urged into politics by his uncle, Judge Roscoe Tartar, he ran for a seat in the Kentucky House of Representatives."},
            {"speaker": "user", "text": "How long was John Sherman Cooper in office?"},
            {"speaker": "agent", "text": "He served as a member of the Kentucky House from 1928-1930."},
            {"speaker": "user", "text": "Did he run for another political office after that?"}
        ],
        "summary": "John Sherman Cooper started his career as a lawyer in Somerset after being admitted to the bar in 1928. He was encouraged by his uncle, Judge Roscoe Tartar, to join politics and ran for the Kentucky House of Representatives as a Republican candidate in 1927. He served in office from 1928-1930.",
        "question": "Did John Sherman Cooper pursue any other political offices after his term in the Kentucky House of Representatives?"
    },
    # Example 2: Sports domain (based on paper's Figure 2 - Martina Hingis)
    {
        "conversation": [
            {"speaker": "user", "text": "Tell me about Martina Hingis's tennis career."},
            {"speaker": "agent", "text": "Martina Hingis is a Swiss former professional tennis player who was ranked world No. 1 multiple times during her career."},
            {"speaker": "user", "text": "Did she have any significant injuries?"},
            {"speaker": "agent", "text": "Yes, she underwent surgeries on both her ankles and struggled with injuries throughout her career."},
            {"speaker": "user", "text": "When did she retire?"},
            {"speaker": "agent", "text": "She retired in 2007 due to a hip injury."},
            {"speaker": "user", "text": "Did she ever come back to play tennis again?"}
        ],
        "summary": "Martina Hingis is a Swiss former professional tennis player who retired in 2007 due to a hip injury. She had previously struggled with injuries, including surgeries on both ankles.",
        "question": "Did Martina Hingis make a return to professional tennis after her 2007 retirement?"
    },
    # Example 3: Technical/Cloud domain
    {
        "conversation": [
            {"speaker": "user", "text": "How do I create a Kubernetes cluster on IBM Cloud?"},
            {"speaker": "agent", "text": "You can create a Kubernetes cluster using the IBM Cloud console, CLI, or API. The basic steps involve selecting a cluster type, configuring worker nodes, and choosing a location."},
            {"speaker": "user", "text": "What types of clusters are available?"},
            {"speaker": "agent", "text": "IBM Cloud offers free clusters for learning and standard clusters for production workloads."},
            {"speaker": "user", "text": "Can I use GPUs in my cluster?"}
        ],
        "summary": "The user is setting up a Kubernetes cluster on IBM Cloud. They've learned about cluster creation methods and the available cluster types including free and standard options.",
        "question": "Does IBM Cloud Kubernetes Service support GPU-enabled worker nodes in clusters?"
    }
]


def get_system_prompt() -> str:
    """
    System prompt for conversation summarization.
    
    Based on SELF-multi-RAG paper Table 13.
    """
    return """You are an expert at summarizing multi-turn conversations for information retrieval.

Your task is to:
1. Summarize the conversation history in 40-50 words, capturing the key context and entities discussed
2. Reformulate the current question so it can be understood independently

The summary and question together should form a self-contained query that can be used to retrieve relevant documents WITHOUT needing the original conversation history.

Guidelines:
- Include important entities, facts, and context from the conversation
- Remove redundant or irrelevant details
- Preserve the user's original intent in the question
- Do NOT introduce new information not present in the conversation
- Keep the total output (summary + question) concise but informative

Output format (EXACTLY this structure):
Summary: [40-50 word summary of conversation context]
Question: [Reformulated standalone question]"""


def format_conversation(messages: list) -> str:
    """Format conversation messages for the prompt."""
    lines = []
    for msg in messages:
        speaker = msg.get("speaker", "user")
        text = msg.get("text", "")
        if speaker == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def format_few_shot_example(example: dict) -> str:
    """Format a single few-shot example."""
    conv_text = format_conversation(example["conversation"])
    return f"""Conversation:
{conv_text}

Summary: {example["summary"]}
Question: {example["question"]}"""


def get_few_shot_prompt() -> str:
    """Build the few-shot examples section."""
    examples = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples.append(f"--- Example {i} ---\n{format_few_shot_example(ex)}")
    return "\n\n".join(examples)


def build_user_prompt(conversation_history: list, current_question: str, include_few_shot: bool = True) -> str:
    """
    Build the complete user prompt for summarization.
    
    Args:
        conversation_history: List of previous messages (user/agent alternating)
        current_question: The current user question to be answered
        include_few_shot: Whether to include few-shot examples
        
    Returns:
        Formatted user prompt string
    """
    parts = []
    
    # Few-shot examples (optional)
    if include_few_shot:
        parts.append("Here are some examples:\n")
        parts.append(get_few_shot_prompt())
        parts.append("\n\n--- Now your turn ---\n")
    
    # Current conversation
    # Combine history with current question
    full_conversation = conversation_history + [{"speaker": "user", "text": current_question}]
    conv_text = format_conversation(full_conversation)
    
    parts.append(f"Conversation:\n{conv_text}")
    parts.append("\nProvide the Summary and Question:")
    
    return "\n".join(parts)


def parse_summary_response(response: str) -> tuple:
    """
    Parse the LLM response to extract summary and question.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Tuple of (summary, question)
    """
    summary = ""
    question = ""
    
    # Clean up response
    response = response.strip()
    
    # Try to parse structured format
    lines = response.split("\n")
    current_field = None
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if line_lower.startswith("summary:"):
            # Save previous field
            if current_field == "question":
                question = " ".join(current_content).strip()
            
            current_field = "summary"
            # Get content after "Summary:"
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            current_content = [content] if content else []
            
        elif line_lower.startswith("question:"):
            # Save previous field
            if current_field == "summary":
                summary = " ".join(current_content).strip()
            
            current_field = "question"
            # Get content after "Question:"
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            current_content = [content] if content else []
            
        elif current_field:
            # Continue current field
            current_content.append(line.strip())
    
    # Save final field
    if current_field == "summary":
        summary = " ".join(current_content).strip()
    elif current_field == "question":
        question = " ".join(current_content).strip()
    
    # Fallback: if parsing failed, try simpler approach
    if not summary or not question:
        if "Summary:" in response and "Question:" in response:
            parts = response.split("Question:")
            summary_part = parts[0].replace("Summary:", "").strip()
            question_part = parts[1].strip() if len(parts) > 1 else ""
            
            summary = summary_part or summary
            question = question_part or question
    
    return summary, question


def format_retrieval_query(summary: str, question: str) -> str:
    """
    Format the summary and question into a retrieval query.
    
    This is what will be used as the query for the retriever.
    """
    return f"Summary: {summary} Question: {question}"

