"""Checkpoint Değerlendirme - Görev 4"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json
from peft import PeftModel
from models.model_loader import load_model_and_tokenizer
from data.dataset_loader import DatasetLoader
from evaluation.evaluator import ModelEvaluator

def evaluate_checkpoint(checkpoint_path, dataset_name, num_samples=None):
    """
    Checkpoint'i değerlendir
    
    Args:
        checkpoint_path: Checkpoint dizini
        dataset_name: "deep" veya "diverse"
        num_samples: Değerlendirilecek örnek sayısı (None = hepsi)
    """
    
    print("=" * 60)
    print(f"Checkpoint Değerlendirme: {checkpoint_path}")
    print("=" * 60)
    
    # 1. Base model yükle
    print("\n1. Base model yükleniyor...")
    base_model, tokenizer = load_model_and_tokenizer(
        use_flash_attention=True,
        load_in_8bit=False
    )
    print("✓ Base model yüklendi")
    
    # 2. LoRA adapter yükle
    print("\n2. LoRA adapter yükleniyor...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    print("✓ LoRA adapter yüklendi")
    
    # 3. Dataset yükle
    print(f"\n3. {dataset_name.upper()} dataset yükleniyor...")
    dataset_loader = DatasetLoader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        use_reasoning=False
    )
    _, test_dataset = dataset_loader.load_and_prepare()
    print(f"✓ Test dataset yüklendi: {len(test_dataset)} örnek")
    
    # 4. Değerlendirme
    print("\n4. Değerlendirme başlıyor...")
    evaluator = ModelEvaluator(model, tokenizer)
    results = evaluator.evaluate_dataset(test_dataset, num_samples=num_samples)
    
    # 5. Sonuçları göster
    print("\n" + "=" * 60)
    print("Değerlendirme Sonuçları")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Test Samples: {results['metrics']['total_samples']}")
    print(f"\nMetrics:")
    for metric, value in results['metrics'].items():
        if metric != 'total_samples':
            print(f"  {metric}: {value:.4f}")
    
    # 6. Sonuçları kaydet
    results_dir = os.path.join(checkpoint_path, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"eval_{dataset_name}.json")
    with open(results_file, "w") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "dataset": dataset_name,
            "metrics": results['metrics']
        }, f, indent=2)
    
    print(f"\n✓ Sonuçlar kaydedildi: {results_file}")
    
    return results


def find_best_checkpoint(base_dir, dataset_name):
    """
    Tüm checkpoint'leri değerlendir ve en iyisini bul
    
    Args:
        base_dir: Checkpoint base dizini (örn: ./checkpoints/deep)
        dataset_name: "deep" veya "diverse"
    """
    
    print("=" * 60)
    print(f"En İyi Checkpoint Bulma: {base_dir}")
    print("=" * 60)
    
    # Checkpoint'leri bul
    checkpoints = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            checkpoints.append(item_path)
    
    if not checkpoints:
        print("Checkpoint bulunamadı!")
        return
    
    print(f"\n{len(checkpoints)} checkpoint bulundu")
    
    # Her checkpoint'i değerlendir
    results = []
    for checkpoint in sorted(checkpoints):
        print(f"\n{'=' * 60}")
        print(f"Değerlendiriliyor: {checkpoint}")
        result = evaluate_checkpoint(checkpoint, dataset_name, num_samples=100)
        results.append({
            "checkpoint": checkpoint,
            "metrics": result['metrics']
        })
    
    # En iyi checkpoint'i bul
    best_checkpoint = max(results, key=lambda x: x['metrics'].get('exact_match', 0))
    
    print("\n" + "=" * 60)
    print("EN İYİ CHECKPOINT")
    print("=" * 60)
    print(f"Checkpoint: {best_checkpoint['checkpoint']}")
    print(f"Metrics:")
    for metric, value in best_checkpoint['metrics'].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint değerlendirme")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint dizini")
    parser.add_argument("--base_dir", type=str, help="Checkpoint base dizini (tüm checkpoint'leri değerlendirmek için)")
    parser.add_argument("--dataset", type=str, required=True, choices=["deep", "diverse"], help="Dataset adı")
    parser.add_argument("--num_samples", type=int, default=None, help="Değerlendirilecek örnek sayısı")
    
    args = parser.parse_args()
    
    if args.checkpoint_path:
        # Tek checkpoint değerlendir
        evaluate_checkpoint(args.checkpoint_path, args.dataset, args.num_samples)
    elif args.base_dir:
        # Tüm checkpoint'leri değerlendir
        find_best_checkpoint(args.base_dir, args.dataset)
    else:
        print("Hata: --checkpoint_path veya --base_dir belirtilmeli")
