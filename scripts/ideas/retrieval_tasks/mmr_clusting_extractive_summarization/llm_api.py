"""
LLM API Functions
Localized LLM calling utilities for Together AI and other providers.
"""

import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def call_together_ai(prompt: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    max_tokens: int = 50, temperature: float = 0.3,
                    max_retries: int = 3, retry_delay: int = 2) -> str:
    """
    Call Together AI API with the given prompt.
    
    Args:
        prompt: The formatted prompt
        model: Model ID on Together AI
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        Generated response text
        
    Raises:
        ValueError: If API key not set
        Exception: If all retry attempts fail
    """
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Format as chat messages
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    
    api_endpoint = "https://api.together.xyz/v1/chat/completions"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            return generated_text.strip()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"API call failed after {max_retries} attempts: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {e}")


def call_llm(prompt: str, llm_config: str = "mixtral", 
            max_tokens: int = 50, temperature: float = 0.3) -> str:
    """
    Call LLM with the given prompt.
    
    Args:
        prompt: The formatted prompt
        llm_config: LLM configuration name (currently only 'mixtral' supported)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    # Map config names to model IDs
    model_map = {
        'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'mixtral-8x7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    }
    
    model = model_map.get(llm_config.lower(), model_map['mixtral'])
    
    return call_together_ai(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )

