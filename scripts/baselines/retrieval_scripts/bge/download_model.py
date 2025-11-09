#!/usr/bin/env python3
"""
Download BGE-base-en-v1.5 model and save it locally.

This script downloads the model to the local models/ directory
so it can be reused without internet access.
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

def download_bge_model():
    """Download BGE-base-en-v1.5 model to local directory."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    model_path = models_dir / "bge-base-en-v1.5"
    
    print("="*80)
    print("BGE Model Download Script")
    print("="*80)
    print("\nModel: BAAI/bge-base-en-v1.5")
    print(f"Save location: {model_path}")
    print("Model size: ~420 MB")
    print()
    
    # Check if model already exists
    if model_path.exists() and (model_path / "config.json").exists():
        print("✅ Model already downloaded!")
        print(f"   Location: {model_path}")
        print("\nTo re-download, delete the models/ directory and run again.")
        return str(model_path)
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📥 Downloading model from HuggingFace...")
        print("   This may take a few minutes (downloading ~420 MB)...")
        print()
        
        # Download model
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        print("💾 Saving model locally...")
        model.save(str(model_path))
        
        print()
        print("="*80)
        print("✅ SUCCESS! Model downloaded and saved")
        print("="*80)
        print(f"Location: {model_path}")
        print(f"Size: {get_dir_size(model_path)}")
        print()
        print("You can now use the model offline by loading from:")
        print(f"  SentenceTransformer('{model_path}')")
        print()
        
        return str(model_path)
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ ERROR: Failed to download model")
        print("="*80)
        print(f"Error: {e}")
        print()
        print("Please check:")
        print("  1. Internet connection is available")
        print("  2. sentence-transformers is installed: pip install sentence-transformers")
        print("  3. Sufficient disk space (~500 MB)")
        print()
        raise

def get_dir_size(path):
    """Get total size of directory in human-readable format."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.1f} TB"

if __name__ == "__main__":
    download_bge_model()

