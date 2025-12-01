"""Evaluation Metrics"""

import numpy as np
from typing import List, Dict

def calculate_pass_at_k(results: List[bool], k: int = 1) -> float:
    """
    Pass@k metriği hesapla
    
    Args:
        results: Her test için başarı durumu (True/False)
        k: Kaç deneme
    
    Returns:
        Pass@k skoru
    """
    n_passed = sum(results)
    n_total = len(results)
    
    if n_total == 0:
        return 0.0
    
    return n_passed / n_total


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Temel metrikleri hesapla
    
    Args:
        predictions: Model çıktıları
        references: Gerçek çözümler
    
    Returns:
        Metrik dictionary
    """
    # Basit exact match
    exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
    
    metrics = {
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "total_samples": len(predictions)
    }
    
    return metrics


def compute_loss_metrics(eval_preds):
    """
    Trainer için loss metriği
    
    Args:
        eval_preds: (predictions, labels) tuple
    
    Returns:
        Metrik dictionary
    """
    predictions, labels = eval_preds
    
    # Loss zaten Trainer tarafından hesaplanıyor
    # Ek metrikler eklenebilir
    
    return {}
