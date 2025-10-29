#!/usr/bin/env python
"""
Step-by-step evaluation runner for MT-RAG generation tasks.
Allows running individual evaluation steps independently.

Usage examples:
  # Run only algorithmic metrics
  python run_step_by_step.py --step algorithmic -i input.jsonl -o output.jsonl
  
  # Run only IDK judge
  python run_step_by_step.py --step idk -i output.jsonl -o output.jsonl --provider hf --judge_model ibm-granite/granite-3.3-8b-instruct
  
  # Run only RAGAS
  python run_step_by_step.py --step ragas -i output.jsonl -o output.jsonl --provider openai --openai_key KEY --azure_host HOST
  
  # Run only RADBench
  python run_step_by_step.py --step radbench -i output.jsonl -o output.jsonl --provider hf --judge_model ibm-granite/granite-3.3-8b-instruct
  
  # Run only IDK conditioning
  python run_step_by_step.py --step idk_condition -i output.jsonl -o output.jsonl
  
  # Run all steps (same as run_generation_eval.py)
  python run_step_by_step.py --step all -i input.jsonl -o output.jsonl --provider hf --judge_model ibm-granite/granite-3.3-8b-instruct
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run individual steps of MT-RAG generation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["algorithmic", "idk", "ragas", "radbench", "idk_condition", "all"],
        help="Which evaluation step to run"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    
    parser.add_argument(
        "-e", "--evaluators",
        type=str,
        default="scripts/evaluation/config.yaml",
        help="Algorithmic evaluators config (for algorithmic step)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "hf"],
        help="Provider for LLM judges (required for idk, ragas, radbench, all)"
    )
    
    parser.add_argument(
        "--judge_model",
        type=str,
        help="HuggingFace model name (required if provider=hf)"
    )
    
    parser.add_argument(
        "--openai_key",
        type=str,
        help="OpenAI API key (required if provider=openai)"
    )
    
    parser.add_argument(
        "--azure_host",
        type=str,
        help="Azure OpenAI endpoint (required if provider=openai)"
    )
    
    args = parser.parse_args()
    
    # Validate provider requirements
    if args.step in ["idk", "ragas", "radbench", "all"]:
        if not args.provider:
            parser.error(f"--step {args.step} requires --provider")
        
        if args.provider == "openai":
            if not args.openai_key or not args.azure_host:
                parser.error("--provider openai requires --openai_key and --azure_host")
            os.environ["AZURE_OPENAI_API_KEY"] = args.openai_key
            os.environ["OPENAI_AZURE_HOST"] = args.azure_host
        elif args.provider == "hf":
            if not args.judge_model:
                parser.error("--provider hf requires --judge_model")
    
    # Run the requested step(s)
    print(f"\n{'='*60}")
    print(f"Running evaluation step: {args.step}")
    print(f"{'='*60}\n")
    
    if args.step == "algorithmic" or args.step == "all":
        print("Step 1/5: Running algorithmic judges (ROUGE, BERTScore, etc.)...")
        from run_algorithmic import run_algorithmic_judges
        run_algorithmic_judges(args.evaluators, args.input, args.output)
        print("✓ Algorithmic metrics complete\n")
    
    if args.step == "idk" or args.step == "all":
        print("Step 2/5: Running IDK judge...")
        from judge_wrapper import run_idk_judge
        model = args.provider if args.provider == "openai" else args.judge_model
        run_idk_judge(model, args.output if args.step == "all" else args.input, args.output)
        print("✓ IDK judge complete\n")
    
    if args.step == "ragas" or args.step == "all":
        print("Step 3/5: Running RAGAS faithfulness judge...")
        from judge_wrapper import run_ragas_judges_openai, run_ragas_judges_local
        if args.provider == "openai":
            run_ragas_judges_openai(
                args.output if args.step == "all" else args.input,
                args.output,
                args.openai_key,
                args.azure_host
            )
        else:
            run_ragas_judges_local(
                args.judge_model,
                args.output if args.step == "all" else args.input,
                args.output
            )
        print("✓ RAGAS complete\n")
    
    if args.step == "radbench" or args.step == "all":
        print("Step 4/5: Running RADBench judge...")
        from judge_wrapper import run_radbench_judge
        model = args.provider if args.provider == "openai" else args.judge_model
        run_radbench_judge(
            model,
            args.output if args.step == "all" else args.input,
            args.output
        )
        print("✓ RADBench complete\n")
    
    if args.step == "idk_condition" or args.step == "all":
        print("Step 5/5: Computing IDK-conditioned metrics...")
        from judge_wrapper import get_idk_conditioned_metrics
        get_idk_conditioned_metrics(args.output, args.output)
        print("✓ IDK conditioning complete\n")
    
    print(f"{'='*60}")
    print(f"✓ Evaluation complete! Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

