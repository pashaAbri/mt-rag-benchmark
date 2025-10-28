"""
Baseline Prompt Strategy

Replicates the exact prompt format from MTRAG paper (Section D.2).
This is the simplest prompting strategy with no special techniques.
"""
from typing import Dict, Any


def construct_prompt(task_data: Dict[str, Any]) -> str:
    """
    Construct the baseline prompt from MTRAG paper.
    
    Paper Section D.2 format:
    - Instruction (generate <150 words, grounded in documents)
    - Passages (numbered PASSAGE 1, 2, ...)
    - Conversation history (User:, Agent:)
    
    Args:
        task_data: Task dictionary with 'contexts' and 'input' fields
        
    Returns:
        Formatted prompt string for LLM
    """
    # Start with instruction (exact text from paper)
    prompt = """Given one or more documents and a user query, generate a response to the query using less than 150 words that is grounded in the provided documents. If no answer can be found in the documents, say, "I do not have specific information"

"""
    
    # Add passages
    contexts = task_data.get('contexts', [])
    for i, context in enumerate(contexts, 1):
        prompt += f"PASSAGE {i}\n"
        prompt += context['text'] + "\n\n"
    
    # Add conversation history
    # Paper format uses capitalized speaker labels: "User:" and "Agent:"
    input_turns = task_data.get('input', [])
    for turn in input_turns:
        speaker = turn['speaker'].capitalize()  # "user" -> "User", "agent" -> "Agent"
        text = turn['text']
        prompt += f"{speaker}: {text}\n"
    
    return prompt


# Metadata for tracking
PROMPT_METADATA = {
    "name": "baseline",
    "description": "Baseline prompt from MTRAG paper (Section D.2)",
    "paper_section": "D.2",
    "techniques": ["simple_instruction"],
    "expected_performance": {
        "llama_3.1_8b": {
            "reference": {"RLF": 0.55, "RBllm": 0.59, "RBalg": 0.36}
        }
    }
}

