from typing import List, Dict, Any
import json
from llm_api import call_llm

class Generator:
    def __init__(self, llm_config: str = 'mixtral'):
        self.llm_config = llm_config
        
        # Template from cleaned_data/conversations/1.json
        # [INST]\n${CONTEXT}\n${SYSTEM_INST}\n${INPUT}\n[/INST]\nanswer:
        self.system_instruction = "You are an AI Assistant, tasked with providing responses that are well-grounded in the provided documents. Given one or more documents and a user query, generate a response to the query. If no answer can be found in the documents, say, \"I do not have specific information\"."
        
    def format_prompt(self, query: str, contexts: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
        """
        Format prompt for generation.
        
        Args:
            query: Current user query
            contexts: List of retrieved documents
            history: Conversation history
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_parts = []
        for doc in contexts:
            text = doc.get('text', '')
            context_parts.append(f"[DOCUMENT]\n{text}\n[END]")
        context_str = "\n".join(context_parts)
        
        # Format history + input
        # Input format: ${SPEAKER}: ${TEXT}
        # We include history here
        input_parts = []
        for turn in history:
            speaker = turn.get('speaker', 'user')
            text = turn.get('text', '')
            input_parts.append(f"{speaker}: {text}")
        
        # Add current turn
        input_parts.append(f"user: {query}")
        input_str = "\n".join(input_parts)
        
        # Assemble full prompt
        prompt = f"[INST]\n{context_str}\n{self.system_instruction}\n{input_str}\n[/INST]\nanswer:"
        return prompt

    def generate(self, query: str, contexts: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
        """
        Generate response.
        
        Args:
            query: User query
            contexts: Retrieved documents
            history: Conversation history
            
        Returns:
            Generated response text
        """
        prompt = self.format_prompt(query, contexts, history)
        
        response = call_llm(
            prompt=prompt,
            llm_config=self.llm_config,
            max_tokens=512,
            temperature=0.7
        )
        return response.strip()

