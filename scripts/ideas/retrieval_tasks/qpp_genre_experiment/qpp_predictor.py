import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

class QPPPredictor:
    def __init__(self, checkpoint_path: str, base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {checkpoint_path} on {self.device}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            if checkpoint_path and os.path.exists(checkpoint_path) and checkpoint_path != base_model_name:
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
                
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def create_prompt(self, query: str, passage: str) -> str:
        return f"Query: {query}\nPassage: {passage}\nRelevant:"

    def predict(self, query: str, passages: List[str]) -> List[int]:
        if not self.model:
            return [0] * len(passages)
            
        preds = []
        for passage in passages:
            prompt = self.create_prompt(query, passage)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=5)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                is_relevant = 1 if "yes" in decoded.lower() or "relevant" in decoded.lower() else 0
                preds.append(is_relevant)
                
        return preds

