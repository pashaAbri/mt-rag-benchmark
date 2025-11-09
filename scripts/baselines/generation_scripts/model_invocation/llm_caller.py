"""
RAG Generation Script

This script runs generation experiments on RAG tasks using LLM APIs.
Follows the prompt format from the MTRAG paper (Section D.2).
"""
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env file
load_dotenv()

# Add current directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_llm_config,
    load_prompt_module,
    get_prompt_id,
    call_together_ai,
    call_openai,
    load_tasks,
    save_results_with_predictions,
    load_existing_results,
    get_completed_task_ids
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run RAG generation experiments with LLMs'
    )
    
    parser.add_argument(
        '--model_config',
        type=str,
        default='llm_configs/llama_3.1_8b.yaml',
        help='Path to model configuration YAML file'
    )
    
    parser.add_argument(
        '--prompt_file',
        type=str,
        default='prompts/baseline.py',
        help='Path to prompt Python module (default: prompts/baseline.py)'
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to generation task file (e.g., human/generation_tasks/RAG.jsonl)'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save results with predictions'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Save checkpoint every N tasks (default: 10)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output file if interrupted'
    )
    
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='Number of concurrent API calls (default: 1)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load model configuration
    print(f"Loading model configuration from {args.model_config}...")
    config = load_llm_config(args.model_config)
    print(f"Model: {config['model_name']}")
    print(f"Provider: {config['provider']}")
    print(f"API Model ID: {config['api_model_id']}")
    
    # Load prompt module
    # Handle both absolute and relative paths
    prompt_file_path = Path(args.prompt_file)
    if not prompt_file_path.is_absolute():
        # If relative, resolve from current working directory (project root)
        prompt_file_path = Path.cwd() / args.prompt_file
    
    print(f"\nLoading prompt module from {prompt_file_path}...")
    prompt_module = load_prompt_module(str(prompt_file_path))
    prompt_id = get_prompt_id(str(prompt_file_path))
    print(f"Prompt ID: {prompt_id}")
    
    # Get prompt metadata if available
    if hasattr(prompt_module, 'PROMPT_METADATA'):
        metadata = prompt_module.PROMPT_METADATA
        print(f"Description: {metadata.get('description', 'N/A')}")
    
    # Get API key based on provider
    provider = config.get('provider', 'together_ai')
    
    if provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("\nError: OPENAI_API_KEY environment variable not set.")
            print("Please add it to your .env file or export it:")
            print("  export OPENAI_API_KEY='your-api-key'")
            sys.exit(1)
    elif provider == 'together_ai':
        api_key = os.environ.get('TOGETHER_API_KEY')
        if not api_key:
            print("\nError: TOGETHER_API_KEY environment variable not set.")
            print("Please add it to your .env file or export it:")
            print("  export TOGETHER_API_KEY='your-api-key'")
            sys.exit(1)
    else:
        print(f"\nError: Unsupported provider '{provider}'")
        print("Supported providers: 'openai', 'together_ai'")
        sys.exit(1)
    
    # Load tasks
    print(f"\nLoading tasks from {args.input_file}...")
    all_tasks = load_tasks(args.input_file)
    print(f"Loaded {len(all_tasks)} tasks")
    
    # Handle resume functionality
    completed_tasks = []
    completed_task_ids = set()
    
    if args.resume:
        print(f"\nChecking for existing results at {args.output_file}...")
        completed_tasks = load_existing_results(args.output_file)
        completed_task_ids = get_completed_task_ids(completed_tasks)
        print(f"Found {len(completed_tasks)} completed tasks")
    
    # Filter out completed tasks
    tasks_to_process = [
        task for task in all_tasks 
        if task['task_id'] not in completed_task_ids
    ]
    
    print(f"\nProcessing {len(tasks_to_process)} tasks...")
    print(f"Batch size: {args.batch_size} (checkpoint every {args.batch_size} tasks)")
    print(f"Concurrency: {args.concurrency} parallel requests")
    print("")
    
    # Process tasks
    processed_count = 0
    all_results = completed_tasks.copy()  # Start with already completed tasks
    results_lock = threading.Lock()  # Thread-safe access to all_results
    
    def process_single_task(task):
        """Process a single task - construct prompt and call API."""
        # Construct prompt using the loaded prompt module's function
        prompt = prompt_module.construct_prompt(task)
        
        # Call API based on provider
        if provider == 'openai':
            generated_text = call_openai(prompt, config, api_key)
        else:  # together_ai
            generated_text = call_together_ai(prompt, config, api_key)
        
        # Add predictions and prompt_id fields while preserving all original fields
        task['predictions'] = [
            {
                "text": generated_text
            }
        ]
        task['prompt_id'] = prompt_id
        
        return task
    
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_single_task, task): task for task in tasks_to_process}
            
            # Process completed tasks
            with tqdm(total=len(tasks_to_process), desc="Generating responses") as pbar:
                for future in as_completed(future_to_task):
                    original_task = future_to_task[future]
                    try:
                        result_task = future.result()
                        
                        with results_lock:
                            all_results.append(result_task)
                            processed_count += 1
                            
                            # Save checkpoint
                            if processed_count % args.batch_size == 0:
                                save_results_with_predictions(all_results, args.output_file)
                                tqdm.write(f"Checkpoint saved: {processed_count}/{len(tasks_to_process)} tasks completed")
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        tqdm.write(f"Error processing task {original_task['task_id']}: {e}")
                        tqdm.write("Saving progress and exiting...")
                        with results_lock:
                            save_results_with_predictions(all_results, args.output_file)
                        sys.exit(1)
        
        # Final save
        save_results_with_predictions(all_results, args.output_file)
        
        print(f"\n✓ Complete! Processed {processed_count} new tasks")
        print(f"  Total results: {len(all_results)} tasks")
        print(f"  Saved to: {args.output_file}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        save_results_with_predictions(all_results, args.output_file)
        print(f"Saved {len(all_results)} tasks to {args.output_file}")
        print("Run with --resume flag to continue from this checkpoint")
        sys.exit(0)


if __name__ == "__main__":
    main()

