"""
Utility functions for RAG generation experiments
"""
import json
import yaml
import time
import requests
from typing import Dict, List, Any
from pathlib import Path


class APICallError(Exception):
    """Exception raised when API call fails after retries."""
    pass


class APIResponseError(Exception):
    """Exception raised when API response format is unexpected."""
    pass


def load_llm_config(config_path: str) -> Dict[str, Any]:
    """
    Load LLM configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompt_module(prompt_file: str):
    """
    Dynamically load a prompt module from file.
    
    Args:
        prompt_file: Path to prompt Python file (e.g., 'prompts/baseline.py')
        
    Returns:
        Loaded module with construct_prompt function
        
    Raises:
        ImportError: If module cannot be loaded
        AttributeError: If module doesn't have construct_prompt function
    """
    import importlib.util
    
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    # Load module from file
    spec = importlib.util.spec_from_file_location("prompt_module", prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Verify it has construct_prompt function
    if not hasattr(module, 'construct_prompt'):
        raise AttributeError(
            f"Prompt module {prompt_file} must have a 'construct_prompt(task_data)' function"
        )
    
    return module


def get_prompt_id(prompt_file: str) -> str:
    """
    Extract prompt ID from prompt filename.
    
    Args:
        prompt_file: Path to prompt file (e.g., 'prompts/baseline.py')
        
    Returns:
        Prompt ID (e.g., 'baseline')
    """
    return Path(prompt_file).stem


def call_together_ai(prompt: str, config: Dict[str, Any], api_key: str, 
                     max_retries: int = 3, retry_delay: int = 2) -> str:
    """
    Call Together AI API with the given prompt.
    
    Args:
        prompt: The formatted prompt
        config: LLM configuration dictionary
        api_key: Together AI API key
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        Generated response text
        
    Raises:
        Exception: If all retry attempts fail
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Format as chat messages for Llama
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    payload = {
        "model": config['api_model_id'],
        "messages": messages,
        "max_tokens": config.get('max_tokens', 200),
        "temperature": config.get('temperature', 0.0),
        "top_p": config.get('top_p', 1.0),
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                config['api_endpoint'],
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            return generated_text
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise APICallError(f"API call failed after {max_retries} attempts: {e}")
        except (KeyError, IndexError) as e:
            raise APIResponseError(f"Unexpected API response format: {e}")


def load_tasks(input_file: str) -> List[Dict[str, Any]]:
    """
    Load tasks from JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        
    Returns:
        List of task dictionaries
    """
    tasks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)
    return tasks


def save_results_with_predictions(tasks: List[Dict[str, Any]], output_file: str):
    """
    Save tasks with predictions to JSONL file.
    
    Args:
        tasks: List of task dictionaries (with predictions field added)
        output_file: Path to output JSONL file
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"Results saved to {output_file}")


def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
    """
    Load existing results from output file for resume functionality.
    
    Args:
        output_file: Path to output JSONL file
        
    Returns:
        List of completed tasks, or empty list if file doesn't exist
    """
    if not Path(output_file).exists():
        return []
    
    return load_tasks(output_file)


def get_completed_task_ids(completed_tasks: List[Dict[str, Any]]) -> set:
    """
    Extract task IDs from completed tasks.
    
    Args:
        completed_tasks: List of completed task dictionaries
        
    Returns:
        Set of completed task IDs
    """
    return {task['task_id'] for task in completed_tasks}

