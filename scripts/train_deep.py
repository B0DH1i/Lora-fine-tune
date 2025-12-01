"""DEEP Dataset ile Training - Görev 3"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.model_loader import load_model_and_tokenizer
from models.lora_setup import setup_lora
from data.dataset_loader import DatasetLoader
from training.trainer import setup_trainer
from config.training_config import TrainingConfig

def train_deep():
    """DEEP dataset ile training"""
    
    print("=" * 60)
    print("DEEP Dataset Training")
    print("=" * 60)
    
    # 1. Model ve tokenizer yükle
    print("\n1. Base model yükleniyor...")
    model, tokenizer = load_model_and_tokenizer(
        use_flash_attention=TrainingConfig.use_flash_attention_2,
        load_in_8bit=TrainingConfig.use_8bit
    )
    print("✓ Model yüklendi")
    
    # 2. LoRA setup
    print("\n2. LoRA yapılandırılıyor...")
    model = setup_lora(model, use_8bit=TrainingConfig.use_8bit)
    print("✓ LoRA yapılandırıldı")
    
    # 3. Dataset yükle
    print("\n3. DEEP dataset yükleniyor...")
    dataset_loader = DatasetLoader(
        dataset_name="deep",
        tokenizer=tokenizer,
        use_reasoning=False  # Sadece solution field
    )
    train_dataset, eval_dataset = dataset_loader.load_and_prepare()
    print(f"✓ Dataset yüklendi - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # 4. Trainer setup
    print("\n4. Trainer yapılandırılıyor...")
    output_dir = "./checkpoints/deep"
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        run_name="deep_training"
    )
    print("✓ Trainer hazır")
    
    # 5. Training başlat
    print("\n5. Training başlıyor...")
    print("=" * 60)
    trainer.train()
    
    # 6. Final model kaydet
    print("\n6. Final model kaydediliyor...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Model kaydedildi: {final_model_path}")
    
    print("\n" + "=" * 60)
    print("DEEP Training tamamlandı!")
    print("=" * 60)
    print(f"\nCheckpoint'ler: {output_dir}")
    print(f"Log'lar: {os.path.join(output_dir, 'logs')}")


if __name__ == "__main__":
    # GPU kontrolü
    if not torch.cuda.is_available():
        print("UYARI: CUDA bulunamadı! CPU'da training çok yavaş olacak.")
        response = input("Devam etmek istiyor musunuz? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    train_deep()
