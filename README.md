# LoRA Fine-Tuning Projesi: Competitive Code Reasoning

Bu proje, Qwen2.5-Coder-1.5B-Instruct modelini LoRA kullanarak iki farklı dataset ile fine-tune etmeyi amaçlar.

## 🚀 Hızlı Başlangıç

```bash
# 1. Kurulum
pip install -r requirements.txt

# 2. Tüm adımları otomatik çalıştır
python scripts/quick_start.py

# VEYA manuel olarak:

# 3. İlk model testi
python scripts/inference_test.py

# 4. Dataset analizi
python scripts/dataset_analysis.py

# 5. Training
python scripts/train_deep.py
python scripts/train_diverse.py

# 6. Değerlendirme
python scripts/evaluate.py --base_dir ./checkpoints/deep --dataset deep
```

## 📁 Proje Yapısı

```
├── config/                      # Konfigürasyon dosyaları
│   ├── training_config.py      # Training hyperparameters
│   └── model_config.py          # Model ve LoRA ayarları
├── data/                        # Dataset işlemleri
│   ├── dataset_loader.py        # Dataset yükleme ve preprocessing
│   └── data_collator.py         # Batch hazırlama
├── models/                      # Model yükleme ve setup
│   ├── model_loader.py          # Base model yükleme
│   └── lora_setup.py            # LoRA konfigürasyonu
├── training/                    # Training loop
│   ├── trainer.py               # Trainer setup
│   └── callbacks.py             # Logging ve early stopping
├── evaluation/                  # Değerlendirme
│   ├── evaluator.py             # Model değerlendirme
│   └── metrics.py               # Metrik hesaplama
├── scripts/                     # Çalıştırılabilir script'ler
│   ├── inference_test.py        # İlk model testi (Görev 1)
│   ├── dataset_analysis.py      # Dataset analizi (Görev 2)
│   ├── train_deep.py            # DEEP training (Görev 3)
│   ├── train_diverse.py         # DIVERSE training (Görev 3)
│   ├── evaluate.py              # Checkpoint değerlendirme (Görev 4)
│   └── quick_start.py           # Tüm adımları çalıştır
├── USAGE_GUIDE.md               # Detaylı kullanım kılavuzu
├── TROUBLESHOOTING.md           # Sorun giderme
├── CHECKLIST.md                 # Teslim kontrol listesi
├── MODEL_CARD_TEMPLATE.md       # HuggingFace model card şablonu
└── requirements.txt             # Gerekli paketler
```



## ⚙️ Konfigürasyon

### Training Hyperparameters
`config/training_config.py` dosyasını düzenleyin:
- Learning rate: `2e-4`
- Batch size: `1` (gradient accumulation: `16`)
- Max epochs: `3`
- Context length: `1024` (solution) / `8192` (reasoning)

### LoRA Konfigürasyonu
`config/model_config.py` dosyasını düzenleyin:
- Rank (r): `32`
- Alpha: `64` (r * 2)
- Dropout: `0.1`
- Target modules: Attention + MLP layers




## 📊 Training Logları

Loglar otomatik kaydedilir:
- Her 20-40 step: train loss
- Her 100-120 step: validation loss
- Konum: `checkpoints/[deep|diverse]/logs/`

## 🔗 Kaynaklar

- **Base Model**: [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- **DEEP Dataset**: [CodeGen-Deep5K](https://huggingface.co/datasets/Naholav/CodeGen-Deep5K)
- **DIVERSE Dataset**: [CodeGen-Diverse-5K](https://huggingface.co/datasets/Naholav/CodeGenDiverse-5K)
- **LoRA Dokümantasyonu**: [HuggingFace LoRA Guide](https://huggingface.co/docs/diffusers/training/lora)



## 📝 Lisans

[Lisans bilgisi eklenecek]
