#!/usr/bin/env python3
"""
Generate formatted markdown tables from aggregated results.

This script reads the aggregated_results.json file (created by calculate_results.py)
and generates formatted markdown tables matching the MT-RAG paper style.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_aggregated_results(results_file: Path) -> Dict[str, Any]:
    """Load pre-calculated aggregated results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def format_value(value: Optional[float], precision: int = 2) -> str:
    """Format a metric value for display."""
    if value is None:
        return "NA"
    return f"{value:.{precision}f}"


def generate_table_5(data: Dict[str, Any], output_dir: Path):
    """Generate Table 5: Generation Results by Retrieval Setting."""
    print("\n=== Generating Table 5: Results by Retrieval Setting ===")
    
    scenarios = data['metadata']['scenarios']
    models = data['metadata']['models']
    model_names = {
        "command_r_plus": "Command-R+ (104B)",
        "gpt_4o": "GPT-4o",
        "gpt_4o_mini": "GPT-4o-mini",
        "llama_3.1_405b": "Llama 3.1 405B Instruct",
        "llama_3.1_70b": "Llama 3.1 70B Instruct",
        "llama_3.1_8b": "Llama 3.1 8B Instruct",
        "mixtral_8x22b": "Mixtral 8x22B Instruct",
        "qwen_2.5_72b": "Qwen 2.5 (72B)",
        "qwen_2.5_7b": "Qwen 2.5 (7B)",
    }
    
    output_file = output_dir / "table_5_generated.md"
    with open(output_file, 'w') as f:
        # Header
        f.write("# Table 5: Generation Results by Retrieval Setting\n\n")
        f.write("<style>\ntable { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n</style>\n\n")
        f.write("**Generated from experimental results**\n\n")
        f.write("Generation results by retrieval setting: Reference (•), Reference+RAG (◐), and RAG (○), ")
        f.write("w/ IDK conditioned metrics. Per column, the best result is in **bold** and second best is <u>underlined</u>.\n\n")
        
        # Table header
        f.write("| | Ans. Acc. | | | | RLF | | | | RBllm | | | | RBalg | | |\n")
        f.write("|-------|-----------|---|---|---|-----|---|---|---|-------|---|---|---|-------|---|---|\n")
        f.write("| | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ |\n")
        
        # Data rows
        for model_key in models:
            model_name = model_names.get(model_key, model_key)
            row = [f"| **{model_name}**"]
            
            # Ans. Acc for each scenario
            for scenario in scenarios:
                scenario_data = data['results'].get(scenario, {})
                model_data = scenario_data.get(model_key)
                if model_data and 'total' in model_data:
                    val = model_data['total'].get('ans_acc')
                    row.append(format_value(val))
                else:
                    row.append("NA")
            row.append("")  # Empty column for spacing
            
            # RLF for each scenario  
            for scenario in scenarios:
                scenario_data = data['results'].get(scenario, {})
                model_data = scenario_data.get(model_key)
                if model_data and 'total' in model_data:
                    val = model_data['total'].get('RL_F_idk')
                    row.append(format_value(val))
                else:
                    row.append("NA")
            row.append("")  # Empty column for spacing
            
            # RBllm for each scenario
            for scenario in scenarios:
                scenario_data = data['results'].get(scenario, {})
                model_data = scenario_data.get(model_key)
                if model_data and 'total' in model_data:
                    val = model_data['total'].get('RB_llm_idk')
                    row.append(format_value(val))
                else:
                    row.append("NA")
            row.append("")  # Empty column for spacing
            
            # RBalg for each scenario
            for scenario in scenarios:
                scenario_data = data['results'].get(scenario, {})
                model_data = scenario_data.get(model_key)
                if model_data and 'total' in model_data:
                    val = model_data['total'].get('RB_agg_idk')
                    row.append(format_value(val))
                else:
                    row.append("NA")
            
            f.write(" | ".join(row) + " |\n")
        
        # Add legend
        f.write("\n## Legend\n\n")
        f.write("- **•** = Reference (perfect retrieval - reference passages only)\n")
        f.write("- **◐** = Reference+RAG (reference passages + top retrieved to reach 5 passages)\n")
        f.write("- **○** = Full RAG (top 5 retrieved passages using Elser with query rewrite)\n")
    
    print(f"  ✓ Generated: {output_file}")


def generate_table_16a(data: Dict[str, Any], output_dir: Path):
    """Generate Table 16a: Results by Answerability."""
    print("\n=== Generating Table 16a: Results by Answerability ===")
    
    models = data['metadata']['models']
    model_names = {
        "command_r_plus": "Command-R+ (104B)",
        "gpt_4o": "GPT-4o",
        "gpt_4o_mini": "GPT-4o-mini",
        "llama_3.1_405b": "Llama 3.1 405B Instruct",
        "llama_3.1_70b": "Llama 3.1 70B Instruct",
        "llama_3.1_8b": "Llama 3.1 8B Instruct",
        "mixtral_8x22b": "Mixtral 8x22B Instruct",
        "qwen_2.5_72b": "Qwen 2.5 (72B)",
        "qwen_2.5_7b": "Qwen 2.5 (7B)",
    }
    
    # Only use reference scenario for this table
    reference_data = data['results'].get('reference', {})
    
    output_file = output_dir / "table_16a_answerability_generated.md"
    with open(output_file, 'w') as f:
        # Header
        f.write("# Table 16a: Generation Results by Answerability\n\n")
        f.write("<style>\ntable { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n</style>\n\n")
        f.write("**Generated from experimental results**\n\n")
        f.write("Detailed generation results in the Reference (•) retrieval setting using three metrics ")
        f.write("(RLF, RBllm, RBalg) broken down by question answerability.\n\n")
        f.write("**Note:** **Bold values** indicate the best-performing model for each metric-answerability combination. ")
        f.write("<u>Underlined values</u> indicate the second-best performing model.\n\n")
        f.write("## Results by Question Answerability\n\n")
        
        # Table header
        f.write("| | Overall | | | | Answerable | | | | Partial | | | | Unans. |\n")
        f.write("|-------|-----|-------|-------|---|-----|-------|-------|---|-----|-------|-------|---|-----|\n")
        f.write("| | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | |\n")
        
        # Data rows
        for model_key in models:
            model_name = model_names.get(model_key, model_key)
            model_data = reference_data.get(model_key)
            
            if not model_data:
                continue
            
            row = [f"| **{model_name}**"]
            
            # Overall
            total = model_data.get('total', {})
            row.append(format_value(total.get('RL_F_idk')))
            row.append(format_value(total.get('RB_llm_idk')))
            row.append(format_value(total.get('RB_agg_idk')))
            row.append("")  # Empty column for spacing
            
            # Answerable
            answerable = model_data.get('by_answerability', {}).get('ANSWERABLE', {})
            row.append(format_value(answerable.get('RL_F_idk')))
            row.append(format_value(answerable.get('RB_llm_idk')))
            row.append(format_value(answerable.get('RB_agg_idk')))
            row.append("")  # Empty column for spacing
            
            # Partially answerable
            partial = model_data.get('by_answerability', {}).get('PARTIAL', {})
            row.append(format_value(partial.get('RL_F_idk')))
            row.append(format_value(partial.get('RB_llm_idk')))
            row.append(format_value(partial.get('RB_agg_idk')))
            row.append("")  # Empty column for spacing
            
            # Unanswerable (just RL_F_idk)
            unanswerable = model_data.get('by_answerability', {}).get('UNANSWERABLE', {})
            row.append(format_value(unanswerable.get('RL_F_idk')))
            
            f.write(" | ".join(row) + " |\n")
    
    print(f"  ✓ Generated: {output_file}")


def generate_table_16b(data: Dict[str, Any], output_dir: Path):
    """Generate Table 16b: Results by Turn Position."""
    print("\n=== Generating Table 16b: Results by Turn Position ===")
    
    models = data['metadata']['models']
    model_names = {
        "command_r_plus": "Command-R+ (104B)",
        "gpt_4o": "GPT-4o",
        "gpt_4o_mini": "GPT-4o-mini",
        "llama_3.1_405b": "Llama 3.1 405B Instruct",
        "llama_3.1_70b": "Llama 3.1 70B Instruct",
        "llama_3.1_8b": "Llama 3.1 8B Instruct",
        "mixtral_8x22b": "Mixtral 8x22B Instruct",
        "qwen_2.5_72b": "Qwen 2.5 (72B)",
        "qwen_2.5_7b": "Qwen 2.5 (7B)",
    }
    
    # Only use reference scenario
    reference_data = data['results'].get('reference', {})
    
    output_file = output_dir / "table_16b_turns_generated.md"
    with open(output_file, 'w') as f:
        # Header
        f.write("# Table 16b: Generation Results by Turn Position\n\n")
        f.write("<style>\ntable { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n</style>\n\n")
        f.write("**Generated from experimental results**\n\n")
        f.write("Detailed generation results in the Reference (•) retrieval setting using three metrics ")
        f.write("(RLF, RBllm, RBalg) broken down by first turn vs subsequent turns.\n\n")
        f.write("**Note:** **Bold values** indicate the best-performing model for each metric-turn combination. ")
        f.write("<u>Underlined values</u> indicate the second-best performing model.\n\n")
        f.write("## Results by Turn Position\n\n")
        
        # Table header
        f.write("| | RLF | | RBllm | | RBalg | |\n")
        f.write("|-------|--------|----------|--------|----------|--------|----------|\n")
        f.write("| | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 |\n")
        
        # Data rows
        for model_key in models:
            model_name = model_names.get(model_key, model_key)
            model_data = reference_data.get(model_key)
            
            if not model_data:
                continue
            
            row = [f"| **{model_name}**"]
            
            turn_1 = model_data.get('by_turn', {}).get('TURN_1', {})
            turn_gt_1 = model_data.get('by_turn', {}).get('TURN_GT_1', {})
            
            # RLF
            row.append(format_value(turn_1.get('RL_F_idk')))
            row.append(format_value(turn_gt_1.get('RL_F_idk')))
            
            # RBllm
            row.append(format_value(turn_1.get('RB_llm_idk')))
            row.append(format_value(turn_gt_1.get('RB_llm_idk')))
            
            # RBalg
            row.append(format_value(turn_1.get('RB_agg_idk')))
            row.append(format_value(turn_gt_1.get('RB_agg_idk')))
            
            f.write(" | ".join(row) + " |\n")
    
    print(f"  ✓ Generated: {output_file}")


def generate_table_16c(data: Dict[str, Any], output_dir: Path):
    """Generate Table 16c: Results by Domain."""
    print("\n=== Generating Table 16c: Results by Domain ===")
    
    models = data['metadata']['models']
    domains = data['metadata']['domains']
    model_names = {
        "command_r_plus": "Command-R+ (104B)",
        "gpt_4o": "GPT-4o",
        "gpt_4o_mini": "GPT-4o-mini",
        "llama_3.1_405b": "Llama 3.1 405B Instruct",
        "llama_3.1_70b": "Llama 3.1 70B Instruct",
        "llama_3.1_8b": "Llama 3.1 8B Instruct",
        "mixtral_8x22b": "Mixtral 8x22B Instruct",
        "qwen_2.5_72b": "Qwen 2.5 (72B)",
        "qwen_2.5_7b": "Qwen 2.5 (7B)",
    }
    
    # Only use reference scenario
    reference_data = data['results'].get('reference', {})
    
    output_file = output_dir / "table_16c_domain_generated.md"
    with open(output_file, 'w') as f:
        # Header
        f.write("# Table 16c: Generation Results by Domain\n\n")
        f.write("<style>\ntable { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n</style>\n\n")
        f.write("**Generated from experimental results**\n\n")
        f.write("Detailed generation results in the Reference (•) retrieval setting using three metrics ")
        f.write("(RLF, RBllm, RBalg) broken down by domain.\n\n")
        f.write("**Note:** **Bold values** indicate the best-performing model for each metric-domain combination. ")
        f.write("<u>Underlined values</u> indicate the second-best performing model.\n\n")
        f.write("## Results by Domain\n\n")
        
        # Table header
        f.write("| | RLF | | | | | RBllm | | | | | RBalg | | | |\n")
        f.write("|-------|---------|------|------|-------|---|---------|------|------|-------|---|---------|------|------|-------|\n")
        f.write("| | CLAPNQ | FiQA | Govt | Cloud | | CLAPNQ | FiQA | Govt | Cloud | | CLAPNQ | FiQA | Govt | Cloud |\n")
        
        # Data rows
        for model_key in models:
            model_name = model_names.get(model_key, model_key)
            model_data = reference_data.get(model_key)
            
            if not model_data:
                continue
            
            row = [f"| **{model_name}**"]
            
            by_domain = model_data.get('by_domain', {})
            
            # RLF for each domain
            for domain in domains:
                domain_data = by_domain.get(domain, {})
                row.append(format_value(domain_data.get('RL_F_idk')))
            row.append("")  # Empty column for spacing
            
            # RBllm for each domain
            for domain in domains:
                domain_data = by_domain.get(domain, {})
                row.append(format_value(domain_data.get('RB_llm_idk')))
            row.append("")  # Empty column for spacing
            
            # RBalg for each domain
            for domain in domains:
                domain_data = by_domain.get(domain, {})
                row.append(format_value(domain_data.get('RB_agg_idk')))
            
            f.write(" | ".join(row) + " |\n")
    
    print(f"  ✓ Generated: {output_file}")


def main():
    """Main function to generate all tables from aggregated results."""
    # Paths - script is in .analysis_generated/, read and write in same directory
    script_dir = Path(__file__).parent  # .analysis_generated/
    output_dir = script_dir  # Output in same directory
    results_file = script_dir / "aggregated_results.json"
    
    print("\n" + "=" * 70)
    print("MT-RAG TABLE GENERATOR")
    print("=" * 70)
    print(f"Reading from: {results_file}")
    print(f"Output directory: {output_dir}")
    
    # Check if aggregated results exist
    if not results_file.exists():
        print("ERROR: Aggregated results file not found!")
        print(f"   Expected: {results_file}")
        print("\n   Please run 'python3 calculate_results.py' first to generate aggregated data.")
        return
    
    # Load aggregated results
    print("\nLoading aggregated results...")
    data = load_aggregated_results(results_file)
    
    scenarios = data['metadata']['scenarios']
    models_count = len(data['metadata']['models'])
    print(f"  ✓ Loaded data for {models_count} models across {len(scenarios)} scenarios")
    
    # Generate all tables
    generate_table_5(data, output_dir)
    generate_table_16a(data, output_dir)
    generate_table_16b(data, output_dir)
    generate_table_16c(data, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ ALL TABLES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nTables saved to: {output_dir}/")
    print("  - table_5_generated.md")
    print("  - table_16a_answerability_generated.md")
    print("  - table_16b_turns_generated.md")
    print("  - table_16c_domain_generated.md")


if __name__ == "__main__":
    main()

