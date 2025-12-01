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
        # Use the same prompt format as QPP-GenRE binary prompt
        return f"Instruction: Please assess the relevance of the provided passage to the following question. Please output \"Relevant\" or \"Irrelevant\".\nQuestion: {query}\nPassage: {passage}\nOutput:"

    def parse_prediction(self, text: str) -> int:
        """Parse the model output to determine relevance (1 for Relevant, 0 for Irrelevant)"""
        # Split on "Output:" to get just the generated part
        if "Output:" in text:
            prediction = text.split("Output:")[-1].strip()
        else:
            prediction = text.strip()
        
        # Check for explicit labels
        prediction_lower = prediction.lower()
        if "relevant" in prediction_lower and "irrelevant" not in prediction_lower:
            return 1
        elif "irrelevant" in prediction_lower or ("ir" in prediction_lower and "relevant" not in prediction_lower):
            return 0
        else:
            # Default to irrelevant if unclear
            return 0

    def predict(self, query: str, passages: List[str]) -> List[int]:
        if not self.model:
            return [0] * len(passages)
            
        preds = []
        for passage in passages:
            prompt = self.create_prompt(query, passage)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                is_relevant = self.parse_prediction(decoded)
                preds.append(is_relevant)
                
        return preds

