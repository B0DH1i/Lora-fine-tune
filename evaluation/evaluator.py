"""Model Evaluation"""

import torch
from tqdm import tqdm
from typing import List, Dict
from evaluation.metrics import calculate_metrics

class ModelEvaluator:
    """Checkpoint değerlendirme"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate_solution(self, problem: str, max_new_tokens: int = 512) -> str:
        """
        Tek bir problem için çözüm üret
        
        Args:
            problem: Problem açıklaması
            max_new_tokens: Maksimum token sayısı
        
        Returns:
            Üretilen çözüm
        """
        # Prompt hazırla
        prompt = f"You are an expert Python programmer. Please read the problem carefully before writing any Python code.\n\nProblem:\n{problem}\n\nSolution:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sadece solution kısmını al
        solution = generated.split("Solution:\n")[-1].strip()
        
        return solution
    
    def evaluate_dataset(self, test_dataset, num_samples: int = None) -> Dict:
        """
        Test dataset üzerinde değerlendirme
        
        Args:
            test_dataset: Test dataset
            num_samples: Değerlendirilecek örnek sayısı (None = hepsi)
        
        Returns:
            Değerlendirme sonuçları
        """
        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
        
        predictions = []
        references = []
        
        print(f"Evaluating {len(test_dataset)} samples...")
        
        for example in tqdm(test_dataset):
            # Problem al (tokenize edilmiş datadan geri çevir)
            # Not: Bu basitleştirilmiş bir yaklaşım
            # Gerçek implementasyonda raw data'yı saklamak daha iyi
            
            try:
                # Çözüm üret
                solution = self.generate_solution(example.get("input", ""))
                predictions.append(solution)
                references.append(example.get("solution", ""))
            except Exception as e:
                print(f"Error generating solution: {e}")
                predictions.append("")
                references.append(example.get("solution", ""))
        
        # Metrikleri hesapla
        metrics = calculate_metrics(predictions, references)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }
