#!/usr/bin/env python3
"""
Generate combined tables showing both Paper and Experimental results side-by-side.
"""

import json
from pathlib import Path
from typing import Dict, Optional


def load_json(file_path: Path) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def format_value(value: Optional[float], precision: int = 2, color: str = None) -> str:
    """Format a metric value for display with optional color."""
    if value is None:
        return "NA"
    
    formatted = f"{value:.{precision}f}"
    
    if color:
        return f'<span class="{color}">{formatted}</span>'
    return formatted


MODEL_NAMES = {
    "command_r_plus": "Command-R+ (104B)",
    "gpt_4o": "GPT-4o",
    "gpt_4o_mini": "GPT-4o-mini",
    "llama_3.1_405b": "Llama 3.1 405B Inst.",
    "llama_3.1_70b": "Llama 3.1 70B Inst.",
    "llama_3.1_8b": "Llama 3.1 8B Inst.",
    "mixtral_8x22b": "Mixtral 8x22B Inst.",
    "qwen_2.5_72b": "Qwen 2.5 (72B)",
    "qwen_2.5_7b": "Qwen 2.5 (7B)",
}


def generate_combined_table_5(paper_data: Dict, exp_data: Dict, output_dir: Path):
    """Generate combined Table 5."""
    print("\n=== Generating Combined Table 5 ===")
    
    output_file = output_dir / "table_5_combined.md"
    
    with open(output_file, 'w') as f:
        f.write("# Table 5: Generation Results by Retrieval Setting (Paper vs. Experimental)\n\n")
        f.write("<style>\n")
        f.write(".paper { color: #0066CC; }  /* Blue for paper results */\n")
        f.write(".exp { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n")
        f.write("</style>\n\n")
        f.write("**Comparison: <span class='paper'>Paper Results (n=426)</span> vs. <span class='exp'>Experimental Results (n=842)</span>**\n\n")
        f.write("Format: <span class='paper'>Paper</span> / <span class='exp'>Experimental</span>\n\n")
        
        # Table header
        f.write("| | Ans. Acc. | | | | RLF | | | | RBllm | | | | RBalg | | |\n")
        f.write("|-------|-----------|---|---|---|-----|---|---|---|-------|---|---|---|-------|---|---|\n")
        f.write("| | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ | | • | ◐ | ○ |\n")
        
        # Data rows
        for model_key in paper_data['metadata']['models']:
            model_name = MODEL_NAMES.get(model_key, model_key)
            row = [f"| **{model_name}**"]
            
            scenarios = ['reference', 'reference_rag', 'full_rag']
            
            # Ans. Acc
            for scenario in scenarios:
                paper_val = paper_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('ans_acc')
                exp_val = exp_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('ans_acc')
                
                if paper_val is not None and exp_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / {format_value(exp_val, color='exp')}")
                elif exp_val is not None:
                    row.append(f"- / {format_value(exp_val, color='exp')}")
                elif paper_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / -")
                else:
                    row.append("NA")
            row.append("")  # Empty column for spacing
            
            # RLF
            for scenario in scenarios:
                paper_val = paper_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('RL_F_idk')
                exp_val = exp_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('RL_F_idk')
                
                if paper_val is not None and exp_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / {format_value(exp_val, color='exp')}")
                elif exp_val is not None:
                    row.append(f"- / {format_value(exp_val, color='exp')}")
                elif paper_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / -")
                else:
                    row.append("NA")
            row.append("")  # Empty column for spacing
            
            # RBllm
            for scenario in scenarios:
                paper_val = paper_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('RB_llm_idk')
                exp_val = exp_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('RB_llm_idk')
                
                if paper_val is not None and exp_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / {format_value(exp_val, color='exp')}")
                elif exp_val is not None:
                    row.append(f"- / {format_value(exp_val, color='exp')}")
                elif paper_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / -")
                else:
                    row.append("NA")
            row.append("")  # Empty column for spacing
            
            # RBalg
            for scenario in scenarios:
                paper_val = paper_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('RB_agg_idk')
                exp_val = exp_data['results'].get(scenario, {}).get(model_key, {}).get('total', {}).get('RB_agg_idk')
                
                if paper_val is not None and exp_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / {format_value(exp_val, color='exp')}")
                elif exp_val is not None:
                    row.append(f"- / {format_value(exp_val, color='exp')}")
                elif paper_val is not None:
                    row.append(f"{format_value(paper_val, color='paper')} / -")
                else:
                    row.append("NA")
            
            f.write(" | ".join(row) + " |\n")
        
        # Legend
        f.write("\n## Legend\n\n")
        f.write("- **•** = Reference (perfect retrieval)\n")
        f.write("- **◐** = Reference+RAG\n")
        f.write("- **○** = Full RAG\n")
        f.write("\n## Notes\n\n")
        f.write("- Paper: n=426 tasks (subset with ≤2 reference passages)\n")
        f.write("- Experimental: n=842 tasks (full dataset)\n")
    
    print(f"  ✓ Generated: {output_file}")


def generate_combined_table_16a(paper_data: Dict, exp_data: Dict, output_dir: Path):
    """Generate combined Table 16a."""
    print("\n=== Generating Combined Table 16a ===")
    
    output_file = output_dir / "table_16a_combined.md"
    
    with open(output_file, 'w') as f:
        f.write("# Table 16a: Results by Answerability (Paper vs. Experimental)\n\n")
        f.write("<style>\n")
        f.write(".paper { color: #0066CC; }  /* Blue for paper results */\n")
        f.write(".exp { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n")
        f.write("</style>\n\n")
        f.write("**Comparison: <span class='paper'>Paper</span> vs. <span class='exp'>Experimental</span> (Reference scenario only)**\n\n")
        f.write("Format: <span class='paper'>Paper</span> / <span class='exp'>Experimental</span>\n\n")
        
        # Table header
        f.write("| | Overall | | | | Answerable | | | | Partial | | | | Unans. |\n")
        f.write("|-------|-----|-------|-------|---|-----|-------|-------|---|-----|-------|-------|---|-----|\n")
        f.write("| | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | RLF | RBllm | RBalg | | |\n")
        
        # Data rows
        paper_ref = paper_data['results'].get('reference', {})
        exp_ref = exp_data['results'].get('reference', {})
        
        for model_key in paper_data['metadata']['models']:
            model_name = MODEL_NAMES.get(model_key, model_key)
            row = [f"| **{model_name}**"]
            
            paper_model = paper_ref.get(model_key, {})
            exp_model = exp_ref.get(model_key, {})
            
            # Overall
            for metric in ['RL_F_idk', 'RB_llm_idk', 'RB_agg_idk']:
                p_val = paper_model.get('total', {}).get(metric)
                e_val = exp_model.get('total', {}).get(metric)
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            row.append("")  # Empty column for spacing
            
            # Answerable
            for metric in ['RL_F_idk', 'RB_llm_idk', 'RB_agg_idk']:
                p_val = paper_model.get('by_answerability', {}).get('ANSWERABLE', {}).get(metric)
                e_val = exp_model.get('by_answerability', {}).get('ANSWERABLE', {}).get(metric)
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            row.append("")  # Empty column for spacing
            
            # Partial
            for metric in ['RL_F_idk', 'RB_llm_idk', 'RB_agg_idk']:
                p_val = paper_model.get('by_answerability', {}).get('PARTIAL', {}).get(metric)
                e_val = exp_model.get('by_answerability', {}).get('PARTIAL', {}).get(metric)
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            row.append("")  # Empty column for spacing
            
            # Unanswerable
            p_val = paper_model.get('by_answerability', {}).get('UNANSWERABLE', {}).get('RL_F_idk')
            e_val = exp_model.get('by_answerability', {}).get('UNANSWERABLE', {}).get('RL_F_idk')
            row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            
            f.write(" | ".join(row) + " |\n")
    
    print(f"  ✓ Generated: {output_file}")


def generate_combined_table_16b(paper_data: Dict, exp_data: Dict, output_dir: Path):
    """Generate combined Table 16b."""
    print("\n=== Generating Combined Table 16b ===")
    
    output_file = output_dir / "table_16b_combined.md"
    
    with open(output_file, 'w') as f:
        f.write("# Table 16b: Results by Turn Position (Paper vs. Experimental)\n\n")
        f.write("<style>\n")
        f.write(".paper { color: #0066CC; }  /* Blue for paper results */\n")
        f.write(".exp { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n")
        f.write("</style>\n\n")
        f.write("**Comparison: <span class='paper'>Paper</span> vs. <span class='exp'>Experimental</span> (Reference scenario only)**\n\n")
        f.write("Format: <span class='paper'>Paper</span> / <span class='exp'>Experimental</span>\n\n")
        
        # Table header
        f.write("| | RLF | | RBllm | | RBalg | |\n")
        f.write("|-------|--------|----------|--------|----------|--------|----------|\n")
        f.write("| | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 | TURN 1 | > TURN 1 |\n")
        
        # Data rows
        paper_ref = paper_data['results'].get('reference', {})
        exp_ref = exp_data['results'].get('reference', {})
        
        for model_key in paper_data['metadata']['models']:
            model_name = MODEL_NAMES.get(model_key, model_key)
            row = [f"| **{model_name}**"]
            
            paper_model = paper_ref.get(model_key, {})
            exp_model = exp_ref.get(model_key, {})
            
            # RLF
            for turn_key in ['TURN_1', 'TURN_GT_1']:
                p_val = paper_model.get('by_turn', {}).get(turn_key, {}).get('RL_F_idk')
                e_val = exp_model.get('by_turn', {}).get(turn_key, {}).get('RL_F_idk')
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            
            # RBllm
            for turn_key in ['TURN_1', 'TURN_GT_1']:
                p_val = paper_model.get('by_turn', {}).get(turn_key, {}).get('RB_llm_idk')
                e_val = exp_model.get('by_turn', {}).get(turn_key, {}).get('RB_llm_idk')
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            
            # RBalg
            for turn_key in ['TURN_1', 'TURN_GT_1']:
                p_val = paper_model.get('by_turn', {}).get(turn_key, {}).get('RB_agg_idk')
                e_val = exp_model.get('by_turn', {}).get(turn_key, {}).get('RB_agg_idk')
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            
            f.write(" | ".join(row) + " |\n")
    
    print(f"  ✓ Generated: {output_file}")


def generate_combined_table_16c(paper_data: Dict, exp_data: Dict, output_dir: Path):
    """Generate combined Table 16c."""
    print("\n=== Generating Combined Table 16c ===")
    
    output_file = output_dir / "table_16c_combined.md"
    domains = paper_data['metadata']['domains']
    
    with open(output_file, 'w') as f:
        f.write("# Table 16c: Results by Domain (Paper vs. Experimental)\n\n")
        f.write("<style>\n")
        f.write(".paper { color: #0066CC; }  /* Blue for paper results */\n")
        f.write(".exp { color: #009900; font-weight: bold; }  /* Green bold for experimental results */\n")
        f.write("</style>\n\n")
        f.write("**Comparison: <span class='paper'>Paper</span> vs. <span class='exp'>Experimental</span> (Reference scenario only)**\n\n")
        f.write("Format: <span class='paper'>Paper</span> / <span class='exp'>Experimental</span>\n\n")
        
        # Table header
        f.write("| | RLF | | | | | RBllm | | | | | RBalg | | | |\n")
        f.write("|-------|---------|------|------|-------|---|---------|------|------|-------|---|---------|------|------|-------|\n")
        f.write("| | CLAPNQ | FiQA | Govt | Cloud | | CLAPNQ | FiQA | Govt | Cloud | | CLAPNQ | FiQA | Govt | Cloud |\n")
        
        # Data rows
        paper_ref = paper_data['results'].get('reference', {})
        exp_ref = exp_data['results'].get('reference', {})
        
        for model_key in paper_data['metadata']['models']:
            model_name = MODEL_NAMES.get(model_key, model_key)
            row = [f"| **{model_name}**"]
            
            paper_model = paper_ref.get(model_key, {})
            exp_model = exp_ref.get(model_key, {})
            
            # RLF for each domain
            for domain in domains:
                p_val = paper_model.get('by_domain', {}).get(domain, {}).get('RL_F_idk')
                e_val = exp_model.get('by_domain', {}).get(domain, {}).get('RL_F_idk')
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            row.append("")  # Empty column for spacing
            
            # RBllm for each domain
            for domain in domains:
                p_val = paper_model.get('by_domain', {}).get(domain, {}).get('RB_llm_idk')
                e_val = exp_model.get('by_domain', {}).get(domain, {}).get('RB_llm_idk')
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            row.append("")  # Empty column for spacing
            
            # RBalg for each domain
            for domain in domains:
                p_val = paper_model.get('by_domain', {}).get(domain, {}).get('RB_agg_idk')
                e_val = exp_model.get('by_domain', {}).get(domain, {}).get('RB_agg_idk')
                row.append(f"{format_value(p_val, color='paper')} / {format_value(e_val, color='exp')}")
            
            f.write(" | ".join(row) + " |\n")
    
    print(f"  ✓ Generated: {output_file}")


def main():
    """Generate combined tables."""
    # Paths
    script_dir = Path(__file__).parent
    
    paper_results_file = script_dir.parent / ".analysis_from_paper" / "paper_results.json"
    exp_results_file = script_dir / "aggregated_results.json"
    output_dir = script_dir
    
    print("=" * 70)
    print("COMBINED TABLE GENERATOR")
    print("=" * 70)
    print(f"Paper results: {paper_results_file}")
    print(f"Experimental results: {exp_results_file}")
    print(f"Output directory: {output_dir}")
    
    # Check files exist
    if not paper_results_file.exists():
        print(f"\nERROR: Paper results not found: {paper_results_file}")
        return
    
    if not exp_results_file.exists():
        print(f"\nERROR: Experimental results not found: {exp_results_file}")
        print("Please run 'python3 calculate_results.py' first")
        return
    
    # Load data
    print("\nLoading data...")
    paper_data = load_json(paper_results_file)
    exp_data = load_json(exp_results_file)
    print("  ✓ Paper data loaded")
    print("  ✓ Experimental data loaded")
    
    # Generate combined tables
    generate_combined_table_5(paper_data, exp_data, output_dir)
    generate_combined_table_16a(paper_data, exp_data, output_dir)
    generate_combined_table_16b(paper_data, exp_data, output_dir)
    generate_combined_table_16c(paper_data, exp_data, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ ALL COMBINED TABLES GENERATED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - table_5_combined.md")
    print("  - table_16a_combined.md")
    print("  - table_16b_combined.md")
    print("  - table_16c_combined.md")


if __name__ == "__main__":
    main()

