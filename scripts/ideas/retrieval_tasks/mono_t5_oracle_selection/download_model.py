#!/usr/bin/env python3
"""
Download and cache the mono-T5 model for local use.
"""

from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Model configuration
MONO_T5_MODEL = "castorini/monot5-base-msmarco"
CACHE_DIR = Path(__file__).parent / ".cache"

def main():
    """Download and cache the mono-T5 model."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading mono-T5 model")
    print("=" * 80)
    print(f"Model: {MONO_T5_MODEL}")
    print(f"Cache directory: {CACHE_DIR}")
    print()
    
    print("Downloading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(
        MONO_T5_MODEL,
        cache_dir=str(CACHE_DIR)
    )
    print("✓ Tokenizer downloaded")
    
    print("Downloading model...")
    model = T5ForConditionalGeneration.from_pretrained(
        MONO_T5_MODEL,
        cache_dir=str(CACHE_DIR)
    )
    print("✓ Model downloaded")
    
    print()
    print("=" * 80)
    print("✓ Model successfully cached!")
    print(f"Cache location: {CACHE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
