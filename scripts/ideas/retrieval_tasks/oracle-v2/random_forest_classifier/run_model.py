#!/usr/bin/env python3
"""
Run Random Forest classifier training and evaluation pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name, args=None):
    """Run a Python script and return success status."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*80}")
    print(f"Running: {script_name} {' '.join(args) if args else ''}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {script_name}: {e}")
        return False


def main():
    scripts = [
        ("train_model.py", []),
        ("evaluate_model.py", []),
    ]
    
    print("="*80)
    print("RANDOM FOREST CLASSIFIER PIPELINE")
    print("="*80)
    
    success_count = 0
    for script_name, args in scripts:
        if run_script(script_name, args):
            success_count += 1
        else:
            print(f"\nPipeline stopped due to failure in {script_name}")
            sys.exit(1)
    
    print("\n" + "="*80)
    print(f"All {success_count}/{len(scripts)} steps completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

