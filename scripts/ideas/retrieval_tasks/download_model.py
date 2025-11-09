"""
Download BGE embedding model for extractive query rewriting.

This script downloads the BAAI/bge-base-en-v1.5 model and saves it locally
to avoid repeated downloads and ensure consistent model versions.
"""

import os
from sentence_transformers import SentenceTransformer
from pathlib import Path


def download_model():
    """Download and cache the BGE model locally."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    models_dir = script_dir / ".models"
    
    # Create hidden models directory
    models_dir.mkdir(exist_ok=True)
    
    model_name = "BAAI/bge-base-en-v1.5"
    local_model_path = models_dir / "bge-base-en-v1.5"
    
    print("=" * 60)
    print("Downloading BGE Embedding Model")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Destination: {local_model_path}")
    print()
    
    if local_model_path.exists():
        print(f"Model already exists at {local_model_path}")
        print("   Skipping download...")
        print()
    else:
        print("Downloading model (this may take a few minutes)...")
        print("   Model size: ~400 MB")
        print()
        
        # Download and save model
        model = SentenceTransformer(model_name)
        model.save(str(local_model_path))
        
        print()
        print("Model downloaded successfully!")
        print()
    
    # Test the model
    print("Testing model...")
    model = SentenceTransformer(str(local_model_path))
    test_embedding = model.encode("test query")
    print(f"   Embedding dimension: {len(test_embedding)}")
    print()
    
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print(f"Model location: {local_model_path}")
    print()
    print("To use this model in your scripts, load it with:")
    print(f'   SentenceTransformer("{local_model_path}")')
    print()
    print("Or use the model name directly:")
    print(f'   SentenceTransformer("{model_name}")')
    print("   (This will use the cached version)")
    print()


if __name__ == "__main__":
    download_model()

