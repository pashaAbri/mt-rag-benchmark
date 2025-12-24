#!/usr/bin/env python3
"""
Run all feature analysis scripts in sequence.
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
        result = subprocess.run(cmd, check=True, capture_output=False)
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
        ("extract_oracle_selections.py", ["--domain", "all"]),
        ("extract_features.py", ["--domain", "all"]),
        ("analyze_patterns.py", []),
        ("train_routing_model.py", ["--data-file", "combined_data.csv"]),
        ("evaluate_routing.py", ["--data-file", "combined_data.csv", "--model-dir", "models"]),
    ]
    
    print("="*80)
    print("FEATURE ANALYSIS PIPELINE")
    print("="*80)
    
    success_count = 0
    for script_name, args in scripts:
        if run_script(script_name, args):
            success_count += 1
        else:
            print(f"\nPipeline stopped due to failure in {script_name}")
            sys.exit(1)
    
    print("\n" + "="*80)
    print(f"All {success_count}/{len(scripts)} scripts completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

