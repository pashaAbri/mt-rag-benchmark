#!/usr/bin/env python3
"""
Inspect tagging disagreements between LLM and Human Annotators.
Focus on "Keyword" classification errors.
"""

import sys
import json
from pathlib import Path

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
TAGGED_DIR = script_dir / "tagged_queries"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

def main():
    print("="*80)
    print("INSPECTING TAGGING DISAGREEMENTS")
    print("="*80)
    
    fp_keyword = [] # Predicted Keyword, but not Human
    fn_keyword = [] # Human Keyword, but not Predicted
    other_mismatch = []
    
    count = 0
    
    for domain in DOMAINS:
        domain_dir = TAGGED_DIR / domain
        if not domain_dir.exists():
            continue
            
        json_files = list(domain_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get Human Tags
                human_tags = []
                if 'user' in data and 'enrichments' in data['user']:
                    enrichments = data['user']['enrichments']
                    if 'Question Type' in enrichments:
                        val = enrichments['Question Type']
                        if isinstance(val, list):
                            human_tags = val
                        else:
                            human_tags = [val]
                
                # Get LLM Tags
                llm_tags = []
                if 'oracle_metadata' in data:
                    val = data['oracle_metadata'].get('predicted_question_type', [])
                    if isinstance(val, list):
                        llm_tags = val
                    elif isinstance(val, str) and val != "Unknown":
                        llm_tags = [val]
                
                # Get Query Text
                query = data.get('user', {}).get('text', "Unknown Query")
                
                if not human_tags or not llm_tags:
                    continue
                    
                # Check for overlap
                overlap = set(human_tags) & set(llm_tags)
                
                if not overlap:
                    # Categorize Error
                    is_pred_keyword = "Keyword" in llm_tags
                    is_human_keyword = "Keyword" in human_tags
                    
                    item = {
                        'query': query,
                        'human': human_tags,
                        'llm': llm_tags,
                        'file': json_file.name
                    }
                    
                    if is_pred_keyword and not is_human_keyword:
                        fp_keyword.append(item)
                    elif is_human_keyword and not is_pred_keyword:
                        fn_keyword.append(item)
                    else:
                        other_mismatch.append(item)
                        
            except Exception:
                continue

    # Print Examples
    print(f"\n--- False Positives: Predicted 'Keyword' (Risky Routing) [{len(fp_keyword)}] ---")
    for i, item in enumerate(fp_keyword[:10]):
        print(f"{i+1}. Query: \"{item['query']}\"")
        print(f"   Human: {item['human']}")
        print(f"   LLM:   {item['llm']}")
        print("-" * 40)

    print(f"\n--- False Negatives: Missed 'Keyword' (Missed Opportunity) [{len(fn_keyword)}] ---")
    for i, item in enumerate(fn_keyword[:10]):
        print(f"{i+1}. Query: \"{item['query']}\"")
        print(f"   Human: {item['human']}")
        print(f"   LLM:   {item['llm']}")
        print("-" * 40)
        
    print(f"\n--- Other Disagreements [{len(other_mismatch)}] ---")
    for i, item in enumerate(other_mismatch[:10]):
        print(f"{i+1}. Query: \"{item['query']}\"")
        print(f"   Human: {item['human']}")
        print(f"   LLM:   {item['llm']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
