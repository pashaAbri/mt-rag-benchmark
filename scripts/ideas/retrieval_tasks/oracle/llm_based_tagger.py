#!/usr/bin/env python3
"""
LLM-Based Context Dependency Tagger

This tagger uses Claude Sonnet 4.5 to determine if a query needs conversation
context (should use 'rewrite' strategy) or is self-contained (can use 'lastturn').

The LLM analyzes linguistic features like:
- Unresolved pronouns (it, this, that, they)
- Implicit references ("the song", "the movie")
- Anaphoric expressions ("what about", "how about")
- Incomplete fragments that require prior context

Usage:
    python llm_based_tagger.py [--workers N] [--limit N]
"""

import os
import sys
import json
import argparse
import concurrent.futures
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


@dataclass
class LLMContextAnalysis:
    """Results from LLM context dependency analysis."""
    needs_context: bool
    confidence: float
    has_unresolved_pronouns: bool
    has_implicit_references: bool
    has_anaphoric_expressions: bool
    is_incomplete_fragment: bool
    reasoning: str
    recommended_strategy: str


# System prompt for context dependency analysis
SYSTEM_PROMPT = """You are an expert at analyzing queries for a multi-turn conversational retrieval system.

Your task is to determine if a query can be understood STANDALONE (self-contained) or if it REQUIRES conversation history to be understood (context-dependent).

## Context-Dependent Indicators (needs history):
1. **Unresolved Pronouns**: "it", "this", "that", "they", "them" without clear referents
   - Example: "How much does it cost?" (What is "it"?)
   - Example: "When was it released?" (What was released?)

2. **Implicit References**: References to things not mentioned in the query
   - Example: "What about the other one?" (Other what?)
   - Example: "Do you know when the song was released?" (Which song?)

3. **Anaphoric Expressions**: Phrases that explicitly reference prior context
   - Example: "What about Romeo and Juliet?" (In context of what topic?)
   - Example: "And how does that work?" (How does what work?)

4. **Incomplete Fragments**: Queries that are grammatically incomplete
   - Example: "How many teams?" (Teams for what?)
   - Example: "Regional differences" (Differences in what?)
   - Example: "Celts" (What about Celts?)

## Self-Contained Indicators (standalone):
1. **Complete Questions**: All entities and context specified
   - Example: "What is the capital of France?"
   - Example: "How do I configure a VPN on macOS?"

2. **Keyword Searches**: Specific lookup terms
   - Example: "Python list sort methods"
   - Example: "AWS S3 pricing"

3. **Troubleshooting with Full Context**: Problem fully described
   - Example: "Error 404 when accessing /api/users endpoint"

## Output Format
Return a JSON object with these fields:
{
    "needs_context": boolean,
    "confidence": float (0.0 to 1.0),
    "has_unresolved_pronouns": boolean,
    "has_implicit_references": boolean,
    "has_anaphoric_expressions": boolean,
    "is_incomplete_fragment": boolean,
    "reasoning": "Brief explanation",
    "recommended_strategy": "rewrite" or "lastturn"
}

IMPORTANT: Return ONLY the JSON object, no other text."""


def get_client() -> anthropic.Anthropic:
    """Initialize Anthropic client."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found. Please set it in .env or environment")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def analyze_query(client: anthropic.Anthropic, query: str, model: str) -> Optional[LLMContextAnalysis]:
    """
    Analyze a single query using the LLM.
    
    Args:
        client: Anthropic client
        query: The user query to analyze
        model: Model ID to use
        
    Returns:
        LLMContextAnalysis or None if error
    """
    try:
        message = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Analyze this query:\n\n\"{query}\""}
            ]
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error for query '{query[:50]}...': {e}")
            return None
        
        # Validate and create dataclass
        # Normalize recommended_strategy (LLM sometimes returns "standalone" instead of "lastturn")
        raw_strategy = result.get('recommended_strategy', 'rewrite').lower()
        if raw_strategy in ['standalone', 'self-contained', 'lastturn']:
            strategy = 'lastturn'
        else:
            strategy = 'rewrite'
        
        return LLMContextAnalysis(
            needs_context=result.get('needs_context', True),
            confidence=float(result.get('confidence', 0.5)),
            has_unresolved_pronouns=result.get('has_unresolved_pronouns', False),
            has_implicit_references=result.get('has_implicit_references', False),
            has_anaphoric_expressions=result.get('has_anaphoric_expressions', False),
            is_incomplete_fragment=result.get('is_incomplete_fragment', False),
            reasoning=result.get('reasoning', ''),
            recommended_strategy=strategy,
        )
        
    except anthropic.APIError as e:
        print(f"API error for query '{query[:50]}...': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for query '{query[:50]}...': {e}")
        return None


def process_file(args_tuple) -> Optional[tuple]:
    """Process a single file (for parallel execution)."""
    input_path, output_path, model = args_tuple
    
    try:
        # Load the file (could be original or already tagged by rule-based)
        source_path = output_path if output_path.exists() else input_path
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if already has LLM tags
        if data.get('oracle_metadata', {}).get('llm_based_tags'):
            return None
        
        # Get query text
        query_text = None
        if 'user' in data:
            query_text = data['user'].get('text') or data['user'].get('utterance')
        
        if not query_text:
            return None
        
        # Create client per call (thread-safe)
        client = get_client()
        
        # Analyze the query
        analysis = analyze_query(client, query_text, model)
        if not analysis:
            return ('error', input_path.name)
        
        # Add oracle_metadata to the data
        if 'oracle_metadata' not in data:
            data['oracle_metadata'] = {}
        
        data['oracle_metadata']['llm_based_tags'] = {
            'tagger': 'llm_sonnet_4.5',
            'model': model,
            **asdict(analysis),
        }
        
        # Write to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return ('success', analysis.needs_context, analysis)
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return ('error', input_path.name)


def main():
    parser = argparse.ArgumentParser(description="LLM-based context dependency tagger")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929",
                        help="Model ID (default: claude-sonnet-4-5-20250929)")
    parser.add_argument("--domains", nargs="+", default=['clapnq', 'cloud', 'fiqa', 'govt'],
                        help="Domains to process")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of queries per domain (for testing)")
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[3]  # mt-rag-benchmark/
    input_dir = project_root / "cleaned_data" / "tasks"
    output_dir = script_dir / "tagged_queries"
    
    # Verify API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found. Please set it in .env or environment")
        sys.exit(1)
    
    print("=" * 60)
    print("LLM-Based Context Dependency Tagger")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Collect all files to process
    all_files = []
    for domain in args.domains:
        domain_input = input_dir / domain
        domain_output = output_dir / domain
        
        if not domain_input.exists():
            print(f"Skipping {domain}: input directory not found")
            continue
        
        json_files = list(domain_input.glob("*.json"))
        if args.limit:
            json_files = json_files[:args.limit]
        
        for f in json_files:
            output_file = domain_output / f.name
            all_files.append((f, output_file, args.model))
        
        print(f"  {domain}: {len(json_files)} files")
    
    print(f"\nTotal files to process: {len(all_files)}")
    
    # Process files in parallel
    total_processed = 0
    needs_context_count = 0
    errors = 0
    feature_counts = {
        'has_unresolved_pronouns': 0,
        'has_implicit_references': 0,
        'has_anaphoric_expressions': 0,
        'is_incomplete_fragment': 0,
    }
    confidences = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, f) for f in all_files]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), desc="Analyzing queries"):
            result = future.result()
            if result is None:
                continue  # Skipped (already exists)
            elif result[0] == 'error':
                errors += 1
            elif result[0] == 'success':
                total_processed += 1
                if result[1]:  # needs_context
                    needs_context_count += 1
                analysis = result[2]
                for feat in feature_counts:
                    if getattr(analysis, feat, False):
                        feature_counts[feat] += 1
                confidences.append(analysis.confidence)
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"Total queries processed: {total_processed}")
    print(f"Errors: {errors}")
    
    if total_processed > 0:
        print(f"\nSummary:")
        print(f"  Needs context (→ rewrite): {needs_context_count} ({needs_context_count/total_processed*100:.1f}%)")
        print(f"  Self-contained (→ lastturn): {total_processed - needs_context_count} ({(total_processed - needs_context_count)/total_processed*100:.1f}%)")
        
        print(f"\nFeature frequencies:")
        for feat, count in feature_counts.items():
            print(f"  {feat}: {count} ({count/total_processed*100:.1f}%)")
        
        if confidences:
            print(f"\nConfidence distribution:")
            print(f"  Mean: {sum(confidences)/len(confidences):.3f}")
            print(f"  High (>0.8): {sum(1 for c in confidences if c > 0.8)} ({sum(1 for c in confidences if c > 0.8)/len(confidences)*100:.1f}%)")
            print(f"  Medium (0.5-0.8): {sum(1 for c in confidences if 0.5 <= c <= 0.8)} ({sum(1 for c in confidences if 0.5 <= c <= 0.8)/len(confidences)*100:.1f}%)")
            print(f"  Low (<0.5): {sum(1 for c in confidences if c < 0.5)} ({sum(1 for c in confidences if c < 0.5)/len(confidences)*100:.1f}%)")


if __name__ == "__main__":
    main()
