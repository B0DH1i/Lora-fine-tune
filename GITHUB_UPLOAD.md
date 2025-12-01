# GitHub'a Yükleme Kılavuzu

## 📦 GitHub'a Gidecek Dosyalar

### ✅ Zorunlu Dosyalar:

```
config/
├── training_config.py
└── model_config.py

models/
├── model_loader.py
└── lora_setup.py

data/
├── dataset_loader.py
└── data_collator.py

training/
├── trainer.py
└── callbacks.py

evaluation/
├── evaluator.py
└── metrics.py

scripts/
├── train_deep.py
├── train_diverse.py
└── evaluate.py

colab_training_deep.ipynb
colab_training_diverse.ipynb
README.md
COLAB_GUIDE.md
requirements.txt
.gitignore
```

### 📝 Opsiyonel:
```
logs/
├── gorev1_base_model_test.md
└── gorev2_dataset_analysis.md
```

### ❌ GİTMEYECEKLER:
```
venv/              # .gitignore'da
__pycache__/       # .gitignore'da
checkpoints/       # Çok büyük
```

## 🚀 Yükleme Adımları

### 1. GitHub Repository Oluştur

1. [github.com](https://github.com) → Sign in
2. Sağ üst **+** → **New repository**
3. Repository name: `lora-finetuning`
4. Public/Private seç
5. **Create repository**

### 2. Lokal Git Başlat

```bash
# Proje dizininde
cd C:\Users\Bodhi\Desktop\lora

# Git başlat
git init

# Dosyaları ekle
git add .

# Commit
git commit -m "LoRA fine-tuning project setup"
```

### 3. GitHub'a Push

```bash
# Remote ekle (YOUR_USERNAME yerine kendi kullanıcı adın)
git remote add origin https://github.com/YOUR_USERNAME/lora-finetuning.git

# Push
git branch -M main
git push -u origin main
```

### 4. Doğrula

GitHub'da repository'ni aç, dosyaların yüklendiğini kontrol et.

## ✅ Kontrol Listesi

Yüklendikten sonra GitHub'da şunlar olmalı:

- [ ] `config/` klasörü
- [ ] `models/` klasörü
- [ ] `data/` klasörü
- [ ] `training/` klasörü
- [ ] `evaluation/` klasörü
- [ ] `scripts/` klasörü
- [ ] `colab_training_deep.ipynb`
- [ ] `colab_training_diverse.ipynb`
- [ ] `README.md`
- [ ] `COLAB_GUIDE.md`
- [ ] `requirements.txt`
- [ ] `.gitignore`

## 🔄 Sonraki Adım

GitHub'a yüklendikten sonra:

1. Repository URL'ini kopyala
2. Colab notebook'larını aç
3. URL'i notebook'ta güncelle:
   ```python
   !git clone https://github.com/YOUR_USERNAME/lora-finetuning.git
   ```
4. Colab'da training'e başla!

---

**Hazır!** GitHub'a yüklendiğinde Colab'da kullanabilirsin 🎉
