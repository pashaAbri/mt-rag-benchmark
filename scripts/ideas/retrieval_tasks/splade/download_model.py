#!/usr/bin/env python3
"""
Download SPLADE model and save it locally.

This script downloads the model to the local models/ directory
so it can be reused without internet access.

Default model: naver/splade-cocondenser-ensembledistil
"""

import os
import argparse
from pathlib import Path


def download_splade_model(model_name: str = "naver/splade-cocondenser-ensembledistil"):
    """Download SPLADE model to local directory."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    
    # Create model-specific directory name
    model_dirname = model_name.replace("/", "-").replace("_", "-")
    model_path = models_dir / model_dirname
    
    print("=" * 80)
    print("SPLADE Model Download Script")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Save location: {model_path}")
    print("Model size: ~440 MB")
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
        print("   This may take a few minutes (downloading ~440 MB)...")
        print()
        
        # Download tokenizer
        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model
        print("   Downloading model weights...")
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        print("💾 Saving model locally...")
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        
        print()
        print("=" * 80)
        print("✅ SUCCESS! Model downloaded and saved")
        print("=" * 80)
        print(f"Location: {model_path}")
        print(f"Size: {get_dir_size(model_path)}")
        print()
        print("You can now use the model offline by loading from:")
        print(f"  AutoModelForMaskedLM.from_pretrained('{model_path}')")
        print()
        
        return str(model_path)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR: Failed to download model")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Please check:")
        print("  1. Internet connection is available")
        print("  2. transformers is installed: pip install transformers")
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
    parser = argparse.ArgumentParser(description='Download SPLADE model')
    parser.add_argument('--model', type=str, 
                        default='naver/splade-cocondenser-ensembledistil',
                        help='HuggingFace model name to download')
    args = parser.parse_args()
    
    download_splade_model(args.model)
