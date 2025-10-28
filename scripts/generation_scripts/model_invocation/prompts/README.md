# Prompt Modules

This directory contains Python-based prompt strategies for RAG generation experiments.

## Format

Each prompt is a **Python module** (`.py` file) that must implement:

### Required Function

```python
def construct_prompt(task_data: Dict[str, Any]) -> str:
    """
    Construct the prompt from task data.
    
    Args:
        task_data: Task dictionary with 'contexts' and 'input' fields
        
    Returns:
        Formatted prompt string for the LLM
    """
    # Your prompting logic here
    return prompt_string
```

### Optional Metadata

```python
PROMPT_METADATA = {
    "name": "prompt_name",
    "description": "Brief description",
    "techniques": ["technique1", "technique2"],
    "expected_performance": {...}
}
```

## Available Prompts

### baseline.py

**Description**: Baseline prompt from MTRAG paper (Section D.2)

**Techniques**: Simple instruction with no special prompting techniques

**Format**:
```
Given one or more documents and a user query, generate a response...

PASSAGE 1
[text]
...
User: [conversation]
Agent: [conversation]
User: [current question]
```

**Use Case**: Replicating paper baseline results

## Creating New Prompts

To create a new prompting strategy:

1. **Create a new `.py` file** (e.g., `few_shot.py`, `chain_of_thought.py`)

2. **Implement `construct_prompt(task_data)` function**:

```python
def construct_prompt(task_data):
    contexts = task_data.get('contexts', [])
    input_turns = task_data.get('input', [])
    
    # Your custom prompt logic here
    prompt = "Your instruction...\n\n"
    
    # Add passages, conversation, etc.
    # You have full control over formatting
    
    return prompt
```

3. **Add metadata** (optional but recommended):

```python
PROMPT_METADATA = {
    "name": "few_shot",
    "description": "Few-shot learning with examples",
    "techniques": ["few_shot", "in_context_learning"]
}
```

4. **Use with `--prompt_file` argument**:

```bash
python model_invocation/llm_caller.py \
    --prompt_file prompts/few_shot.py \
    ...
```

## Advanced Prompting Examples

### Few-Shot Learning

```python
def construct_prompt(task_data):
    prompt = "Here are examples:\n\n"
    prompt += "Example 1: ...\n"
    prompt += "Example 2: ...\n\n"
    prompt += "Now answer:\n\n"
    # Add passages and conversation
    return prompt
```

### Chain-of-Thought

```python
def construct_prompt(task_data):
    prompt = "Let's think step by step:\n\n"
    # Add passages and conversation
    prompt += "Reasoning: Think through the answer before responding.\n"
    return prompt
```

### Conditional Logic

```python
def construct_prompt(task_data):
    answerability = task_data.get('Answerability', ['ANSWERABLE'])[0]
    
    if answerability == 'UNANSWERABLE':
        # Different instruction for unanswerable questions
        prompt = "Carefully check if the answer is in the documents...\n"
    else:
        prompt = "Standard instruction...\n"
    
    # Add passages and conversation
    return prompt
```

## Benefits of Python-Based Prompts

- **Flexibility**: Insert data anywhere, use conditionals, loops
- **Logic**: Different prompts based on task properties
- **Reusability**: Share helper functions across prompts
- **Version Control**: Track prompt changes in git
- **Testing**: Unit test prompt construction
- **Documentation**: Code comments explain reasoning

## Prompt ID Tracking

The prompt filename (without `.py` extension) becomes the `prompt_id` in results:

- `baseline.py` → `"prompt_id": "baseline"`
- `few_shot.py` → `"prompt_id": "few_shot"`
- `chain_of_thought.py` → `"prompt_id": "chain_of_thought"`

This allows comparing performance across different prompting strategies.
