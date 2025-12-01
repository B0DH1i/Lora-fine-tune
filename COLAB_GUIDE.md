# Google Colab Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### Adım 1: Projeyi GitHub'a Yükle

```bash
# Lokal bilgisayarında
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/lora-finetuning.git
git push -u origin main
```

### Adım 2: Google Colab'ı Aç

1. [colab.research.google.com](https://colab.research.google.com) adresine git
2. **File > Upload notebook** seç
3. `colab_training_deep.ipynb` dosyasını yükle

### Adım 3: GPU'yu Aktif Et

1. **Runtime > Change runtime type** tıkla
2. **Hardware accelerator**: T4 GPU seç
3. **Save** tıkla

### Adım 4: Notebook'u Çalıştır

Her hücreyi sırayla çalıştır (Shift+Enter):

1. ✅ GPU kontrolü
2. ✅ Paketleri kur
3. ✅ Projeyi indir
4. ✅ Drive'ı bağla
5. ✅ Config ayarla
6. ✅ Model yükle
7. ✅ Training başlat (2-4 saat)
8. ✅ Model kaydet

## 📋 Detaylı Adımlar

### 1. GPU Kontrolü

```python
!nvidia-smi
```

**Beklenen çıktı**: Tesla T4, 16GB VRAM

### 2. Paket Kurulumu

```python
!pip install -q torch transformers peft datasets accelerate bitsandbytes tqdm
```

**Süre**: ~2-3 dakika

### 3. Proje İndirme

```python
!git clone https://github.com/YOUR_USERNAME/lora-finetuning.git
%cd lora-finetuning
```

**Not**: `YOUR_USERNAME` yerine kendi GitHub kullanıcı adını yaz!

### 4. Google Drive Bağlantısı

```python
from google.colab import drive
drive.mount('/content/drive')
```

**İzin ver**: Google hesabını seç ve izin ver

**Neden gerekli?**: 
- Checkpoint'ler Drive'a kaydedilir
- Oturum bitince silinmez
- Sonra indirebilirsin

### 5. Training

Training başladığında:
- ⏱️ **Süre**: 2-4 saat
- 📊 **İlerleme**: Progress bar göreceksin
- 💾 **Otomatik kayıt**: Her 100 step'te checkpoint kaydedilir
- ⚠️ **Oturumu açık tut**: Tarayıcıyı kapatma!

### 6. Training Sırasında İzleme

```python
# Loss değerlerini görmek için
# Her 20-40 step'te train loss
# Her 100-120 step'te eval loss
```

## 🔄 DIVERSE Dataset için

1. `colab_training_diverse.ipynb` dosyasını yükle
2. Aynı adımları tekrarla
3. **Önemli**: DEEP training bittikten sonra başla!

## 💾 Checkpoint'leri İndirme

### Yöntem 1: Drive'dan İndir

1. Google Drive'ı aç
2. `MyDrive/lora_checkpoints/` klasörüne git
3. `deep/` veya `diverse/` klasörünü indir

### Yöntem 2: Notebook'tan İndir

```python
# Log dosyalarını zip'le
!zip -r training_logs.zip /content/drive/MyDrive/lora_checkpoints/deep/logs

# İndir
from google.colab import files
files.download('training_logs.zip')
```

## ⚠️ Önemli Notlar

### Oturum Kesilirse

Eğer oturum kesilirse (12 saat veya 90 dakika boşta):
1. ✅ **Checkpoint'ler Drive'da kayıtlı** - Kaybolmaz!
2. ✅ Training'e kaldığı yerden devam edebilirsin
3. ✅ En son checkpoint'ten devam et

### Devam Etme Kodu

```python
# En son checkpoint'i bul
import os
checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
resume_from = os.path.join(checkpoint_dir, latest_checkpoint)

# Training'e devam et
trainer.train(resume_from_checkpoint=resume_from)
```

### Memory Sorunları

Eğer OOM (Out of Memory) hatası alırsan:

```python
# Config'i güncelle
TrainingConfig.per_device_batch_size = 1  # Zaten 1
TrainingConfig.gradient_accumulation_steps = 32  # 16'dan artır
TrainingConfig.max_length_solution = 800  # 1024'ten düşür
```

## 📊 Training Sonrası

### 1. Log'ları İncele

```python
import json

log_file = '/content/drive/MyDrive/lora_checkpoints/deep/logs/training_log_*.jsonl'

# Log'ları oku
with open(log_file, 'r') as f:
    logs = [json.loads(line) for line in f]

# Train loss'ları göster
train_losses = [(log['step'], log['train_loss']) for log in logs if 'train_loss' in log]
print(train_losses)
```

### 2. Model Test Et

```python
# Eğitilmiş model ile test
test_problem = "Write a function to reverse a string"
prompt = f"You are an expert Python programmer. Please read the problem carefully before writing any Python code.\n\nProblem:\n{test_problem}\n\nSolution:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(solution)
```

### 3. HuggingFace'e Yükle

```python
# HuggingFace login
from huggingface_hub import notebook_login
notebook_login()

# Model yükle
model.push_to_hub("your-username/qwen-coder-lora-deep")
tokenizer.push_to_hub("your-username/qwen-coder-lora-deep")
```

## 🎯 Checklist

### DEEP Training:
- [ ] Colab'da GPU aktif
- [ ] Notebook yüklendi
- [ ] GitHub repo linki güncellendi
- [ ] Drive bağlandı
- [ ] Training başladı
- [ ] Training tamamlandı (2-4 saat)
- [ ] Checkpoint'ler Drive'a kaydedildi
- [ ] Log'lar indirildi

### DIVERSE Training:
- [ ] DEEP training bitti
- [ ] Yeni Colab oturumu açıldı
- [ ] GPU aktif
- [ ] `colab_training_diverse.ipynb` yüklendi
- [ ] Training başladı
- [ ] Training tamamlandı
- [ ] Checkpoint'ler kaydedildi

## 💡 İpuçları

1. **Oturumu açık tut**: Tarayıcı sekmesini kapatma
2. **İnternet bağlantısı**: Stabil olmalı
3. **Drive alanı**: ~10-15GB boş alan gerekli
4. **Gece çalıştır**: Uzun süreceği için gece başlat
5. **İki training ayrı**: DEEP ve DIVERSE'i ayrı oturumlarda yap

## 🆘 Sorun Giderme

### "Runtime disconnected"
- Oturum kesildi, checkpoint'ten devam et
- Drive'daki checkpoint'ler kayıtlı

### "CUDA out of memory"
- Batch size'ı düşür (zaten 1)
- Context length'i düşür (800'e)
- Runtime'ı restart et

### "Dataset not found"
- İnternet bağlantısını kontrol et
- HuggingFace erişilebilir mi kontrol et

### "Module not found"
- Paketleri tekrar kur
- Proje dizinini kontrol et (`%cd lora-finetuning`)

## 📞 Yardım

Sorun yaşarsan:
1. Hata mesajını kaydet
2. Hangi hücrede olduğunu not et
3. GPU durumunu kontrol et (`!nvidia-smi`)
4. E-posta: arda.mulayim@outlook.com

---

**Başarılar!** 🚀
