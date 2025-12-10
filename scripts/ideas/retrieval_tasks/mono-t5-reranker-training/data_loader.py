#!/usr/bin/env python3
"""
Utility functions for loading actual text data from references at training time.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

# Hardcoded paths
PROJECT_ROOT = Path('/Users/pastil/Dev/Github/mt-rag-benchmark')
CORPUS_LEVEL = 'passage_level'  # or 'document_level'


class DataLoader:
    """Loads query and document text from original data sources."""
    
    def __init__(self, corpus_level: str = CORPUS_LEVEL):
        self.corpus_level = corpus_level
        self._queries_cache: Dict[str, Dict[str, str]] = defaultdict(dict)  # domain_strategy -> query_id -> text
        self._corpus_cache: Dict[str, Dict[str, str]] = {}  # domain -> doc_id -> text
        self._retrieval_cache: Dict[str, Dict[str, str]] = defaultdict(dict)  # domain_strategy -> doc_id -> text
        
    def load_query_text(self, query_id: str, domain: str, strategy: str) -> Optional[str]:
        """Load query text from original query file."""
        cache_key = f"{domain}_{strategy}"
        
        if cache_key not in self._queries_cache:
            queries_path = PROJECT_ROOT / 'human' / 'retrieval_tasks' / domain / f'{domain}_{strategy}.jsonl'
            if queries_path.exists():
                with open(queries_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        qid = data.get('_id')
                        qtext = data.get('text', '')
                        if qid:
                            self._queries_cache[cache_key][qid] = qtext
        
        return self._queries_cache[cache_key].get(query_id)
    
    def load_document_text(self, document_id: str, domain: str, strategy: str) -> Optional[str]:
        """Load document text from corpus or retrieval results."""
        # Try corpus first
        if domain not in self._corpus_cache:
            corpus_path = PROJECT_ROOT / 'corpora' / self.corpus_level / f'{domain}.jsonl'
            if corpus_path.exists():
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        doc_id = data.get('_id')
                        doc_text = data.get('text', '') or data.get('contents', '') or data.get('body', '')
                        if doc_id and doc_text:
                            self._corpus_cache[domain][doc_id] = doc_text
        
        if document_id in self._corpus_cache.get(domain, {}):
            return self._corpus_cache[domain][document_id]
        
        # Fallback: try retrieval results
        cache_key = f"{domain}_{strategy}"
        if cache_key not in self._retrieval_cache:
            results_path = PROJECT_ROOT / 'scripts' / 'baselines' / 'retrieval_scripts' / 'elser' / 'results' / f'elser_{domain}_{strategy}_evaluated.jsonl'
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        contexts = data.get('contexts', [])
                        for ctx in contexts:
                            doc_id = ctx.get('document_id')
                            doc_text = ctx.get('text', '')
                            if doc_id and doc_text:
                                self._retrieval_cache[cache_key][doc_id] = doc_text
        
        return self._retrieval_cache[cache_key].get(document_id)
    
    def format_monot5_example(self, query_id: str, document_id: str, domain: str, strategy: str) -> Optional[Dict[str, str]]:
        """
        Format an example in MonoT5 format.
        
        Returns:
            Dict with 'input' and 'output' keys, or None if data not found
        """
        query_text = self.load_query_text(query_id, domain, strategy)
        doc_text = self.load_document_text(document_id, domain, strategy)
        
        if not query_text or not doc_text:
            return None
        
        input_text = f"Query: {query_text} Document: {doc_text} Relevant:"
        
        return {
            'input': input_text,
            'query_id': query_id,
            'document_id': document_id,
            'domain': domain,
            'strategy': strategy
        }


def load_example_references(data_file: Path):
    """Load example references from JSONL file."""
    examples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples
