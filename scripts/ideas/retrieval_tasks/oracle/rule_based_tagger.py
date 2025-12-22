#!/usr/bin/env python3
"""
Rule-Based Context Dependency Tagger

This tagger uses linguistic patterns to determine if a query needs conversation
context (should use 'rewrite' strategy) or is self-contained (can use 'lastturn').

Based on analysis findings:
- Queries with pronouns (it, this, that, they) often need context
- Short fragments (≤4 words) often need context
- Implicit references ("the song", "the movie") need context
- Anaphoric expressions ("what about", "how about") need context

Usage:
    python rule_based_tagger.py [--domains DOMAIN1 DOMAIN2 ...]
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from tqdm import tqdm


@dataclass
class ContextDependencyTags:
    """Tags indicating context dependency characteristics."""
    is_fragment: bool = False
    has_pronoun: bool = False
    has_demonstrative: bool = False
    has_implicit_reference: bool = False
    has_anaphoric_expression: bool = False
    has_ellipsis: bool = False
    word_count: int = 0
    char_count: int = 0
    
    # Derived predictions
    needs_context: bool = False
    confidence: float = 0.0
    recommended_strategy: str = "rewrite"
    matched_patterns: List[str] = None
    
    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []


class RuleBasedTagger:
    """
    Fast, deterministic query tagging using regex patterns.
    
    Patterns are derived from analysis of 116 queries where rewrite genuinely
    beats lastturn, identifying linguistic markers of context dependency.
    """
    
    # Pronouns that often reference prior context
    PRONOUNS = {'it', 'its', 'itself'}
    
    # Demonstratives that reference prior context
    DEMONSTRATIVES = {'this', 'that', 'these', 'those'}
    
    # Third-person pronouns
    THIRD_PERSON = {'they', 'them', 'their', 'theirs', 'he', 'him', 'his', 'she', 'her', 'hers'}
    
    # Implicit reference patterns (compile once for efficiency)
    IMPLICIT_REF_PATTERNS = [
        re.compile(r'\bthe (song|movie|book|game|team|company|product|article|document|file|system|service|feature|option|method|function|error|issue|problem|solution)\b', re.IGNORECASE),
        re.compile(r'\b(that|this) (one|thing|item|option|approach|method|way)\b', re.IGNORECASE),
        re.compile(r'\bthe (same|other|first|second|third|last|next|previous)\b', re.IGNORECASE),
    ]
    
    # Anaphoric expression patterns
    ANAPHORIC_PATTERNS = [
        re.compile(r'\b(what|how) about\b', re.IGNORECASE),
        re.compile(r'^and (what|how|when|where|who|why)\b', re.IGNORECASE),
        re.compile(r'\b(mentioned|said|discussed|talked about|referred to|described)\b', re.IGNORECASE),
        re.compile(r'\b(earlier|previous|above|before|prior)\b', re.IGNORECASE),
        re.compile(r'\balso\b', re.IGNORECASE),
        re.compile(r'^(also|and|but|so|then)\b', re.IGNORECASE),
    ]
    
    # Ellipsis patterns (incomplete sentences)
    ELLIPSIS_PATTERNS = [
        re.compile(r'^(why|how|what|when|where|who)\s*\??\s*$', re.IGNORECASE),  # Single word questions
        re.compile(r'^(any|more|other|another)\s+\w+\s*\??\s*$', re.IGNORECASE),  # "Any X?"
    ]
    
    def tag(self, query: str) -> ContextDependencyTags:
        """
        Analyze a query and return context dependency tags.
        
        Args:
            query: The user query text
            
        Returns:
            ContextDependencyTags with analysis results
        """
        query = query.strip()
        query_lower = query.lower()
        words = query_lower.split()
        word_set = set(words)
        
        tags = ContextDependencyTags(
            word_count=len(words),
            char_count=len(query),
            matched_patterns=[],
        )
        
        # Check for fragment (short query)
        if len(words) <= 4 or len(query) <= 25:
            tags.is_fragment = True
            tags.matched_patterns.append("fragment")
        
        # Check for pronouns
        if word_set & self.PRONOUNS:
            tags.has_pronoun = True
            matched = word_set & self.PRONOUNS
            tags.matched_patterns.append(f"pronoun:{','.join(matched)}")
        
        # Check for demonstratives
        if word_set & self.DEMONSTRATIVES:
            tags.has_demonstrative = True
            matched = word_set & self.DEMONSTRATIVES
            tags.matched_patterns.append(f"demonstrative:{','.join(matched)}")
        
        # Check for third-person pronouns
        if word_set & self.THIRD_PERSON:
            tags.has_pronoun = True
            matched = word_set & self.THIRD_PERSON
            tags.matched_patterns.append(f"third_person:{','.join(matched)}")
        
        # Check for implicit references
        for pattern in self.IMPLICIT_REF_PATTERNS:
            match = pattern.search(query_lower)
            if match:
                tags.has_implicit_reference = True
                tags.matched_patterns.append(f"implicit_ref:{match.group()}")
                break
        
        # Check for anaphoric expressions
        for pattern in self.ANAPHORIC_PATTERNS:
            match = pattern.search(query_lower)
            if match:
                tags.has_anaphoric_expression = True
                tags.matched_patterns.append(f"anaphoric:{match.group()}")
                break
        
        # Check for ellipsis
        for pattern in self.ELLIPSIS_PATTERNS:
            if pattern.match(query_lower):
                tags.has_ellipsis = True
                tags.matched_patterns.append("ellipsis")
                break
        
        # Calculate confidence and prediction
        tags.needs_context, tags.confidence = self._calculate_prediction(tags)
        tags.recommended_strategy = "rewrite" if tags.needs_context else "lastturn"
        
        return tags
    
    def _calculate_prediction(self, tags: ContextDependencyTags) -> tuple:
        """
        Calculate whether query needs context based on tags.
        
        Returns:
            (needs_context: bool, confidence: float)
        """
        # Count context dependency signals
        signals = 0
        weights = 0.0
        
        # High-confidence signals (from analysis: 3.39x lift)
        if tags.has_implicit_reference:
            signals += 1
            weights += 0.35
        
        # Medium-high signals (1.9x lift)
        if tags.is_fragment and tags.has_pronoun:
            signals += 1
            weights += 0.30
        elif tags.is_fragment:
            signals += 1
            weights += 0.15
        elif tags.has_pronoun:
            signals += 1
            weights += 0.15
        
        # Medium signals (1.7x lift)
        if tags.has_demonstrative:
            signals += 1
            weights += 0.15
        
        # Anaphoric expressions
        if tags.has_anaphoric_expression:
            signals += 1
            weights += 0.20
        
        # Ellipsis
        if tags.has_ellipsis:
            signals += 1
            weights += 0.25
        
        # Calculate confidence
        # Base confidence starts at 0.5, adjust based on signals
        if signals == 0:
            return False, 0.7  # Confident it's self-contained
        elif signals == 1:
            return True, 0.5 + weights
        else:
            return True, min(0.95, 0.5 + weights)


def main():
    parser = argparse.ArgumentParser(description="Rule-based context dependency tagger")
    parser.add_argument("--domains", nargs="+", default=['clapnq', 'cloud', 'fiqa', 'govt'],
                        help="Domains to process")
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[3]  # mt-rag-benchmark/
    input_dir = project_root / "cleaned_data" / "tasks"
    output_dir = script_dir / "tagged_queries"
    
    # Initialize tagger
    tagger = RuleBasedTagger()
    
    # Stats
    total_processed = 0
    needs_context_count = 0
    pattern_counts = {}
    
    print("=" * 60)
    print("Rule-Based Context Dependency Tagger")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    for domain in args.domains:
        domain_input = input_dir / domain
        domain_output = output_dir / domain
        
        if not domain_input.exists():
            print(f"\nSkipping {domain}: input directory not found")
            continue
        
        # Create output directory
        domain_output.mkdir(parents=True, exist_ok=True)
        
        json_files = list(domain_input.glob("*.json"))
        print(f"\nProcessing {domain}: {len(json_files)} files")
        
        for filepath in tqdm(json_files, desc=f"Tagging {domain}"):
            try:
                # Load original file
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get query text
                query_text = None
                if 'user' in data:
                    query_text = data['user'].get('text') or data['user'].get('utterance')
                
                if not query_text:
                    continue
                
                # Tag the query
                tags = tagger.tag(query_text)
                
                # Add oracle_metadata to the data
                if 'oracle_metadata' not in data:
                    data['oracle_metadata'] = {}
                
                data['oracle_metadata']['rule_based_tags'] = {
                    'tagger': 'rule_based_v1',
                    'needs_context': tags.needs_context,
                    'confidence': tags.confidence,
                    'recommended_strategy': tags.recommended_strategy,
                    'is_fragment': tags.is_fragment,
                    'has_pronoun': tags.has_pronoun,
                    'has_demonstrative': tags.has_demonstrative,
                    'has_implicit_reference': tags.has_implicit_reference,
                    'has_anaphoric_expression': tags.has_anaphoric_expression,
                    'has_ellipsis': tags.has_ellipsis,
                    'word_count': tags.word_count,
                    'char_count': tags.char_count,
                    'matched_patterns': tags.matched_patterns,
                }
                
                # Write to output
                output_file = domain_output / filepath.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Update stats
                total_processed += 1
                if tags.needs_context:
                    needs_context_count += 1
                for p in tags.matched_patterns:
                    pattern_type = p.split(':')[0]
                    pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
                    
            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"Total queries tagged: {total_processed}")
    
    if total_processed > 0:
        print(f"\nSummary:")
        print(f"  Needs context (→ rewrite): {needs_context_count} ({needs_context_count/total_processed*100:.1f}%)")
        print(f"  Self-contained (→ lastturn): {total_processed - needs_context_count} ({(total_processed - needs_context_count)/total_processed*100:.1f}%)")
        
        print(f"\nPattern frequencies:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count} ({count/total_processed*100:.1f}%)")


if __name__ == "__main__":
    main()
