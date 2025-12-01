"""Data Collator for Batching"""

from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class DataCollatorForCausalLM:
    """Batch hazırlama için collator"""
    
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Batch oluştur
        
        Args:
            features: Tokenize edilmiş örnekler listesi
        
        Returns:
            Batch dictionary
        """
        # Input IDs
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        
        # Tensor'a çevir
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
        
        return batch
