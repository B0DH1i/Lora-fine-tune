# 🚀 Demo Arayüzü Kurulum

## Özellikler:

✨ **2 Model Karşılaştırma**: DEEP vs DIVERSE  
⚙️ **Ayarlanabilir Parametreler**: Temperature, Top P, Max Tokens  
🎯 **Tek Model Modu**: Bir model seç, test et  
⚖️ **Karşılaştırma Modu**: İki modeli yan yana test et  
📚 **Örnek Problemler**: Hazır test soruları  
🎨 **Modern Arayüz**: Gradio ile profesyonel tasarım  

---

## 🔧 Kurulum:

### 1. Lokal Bilgisayarda:

```bash
# Paketleri kur
pip install -r demo_requirements.txt

# Çalıştır
python demo_app.py
```

Tarayıcıda açılacak: `http://localhost:7860`

### 2. HuggingFace Space'te (Önerilen):

#### Adım 1: Space Oluştur
1. [huggingface.co/spaces](https://huggingface.co/spaces) git
2. **Create new Space** tıkla
3. **Space name**: `qwen-coder-lora-demo`
4. **SDK**: **Gradio** seç
5. **Create Space**

#### Adım 2: Dosyaları Yükle
Space'e şu dosyaları yükle:
- `app.py` (demo_app.py'yi yeniden adlandır)
- `requirements.txt` (demo_requirements.txt'yi yeniden adlandır)

#### Adım 3: Model Linklerini Güncelle
`app.py` içinde:
```python
# Satır 23-24
models["DEEP"] = PeftModel.from_pretrained(
    base_model,
    "B0DH1i/qwen-coder-lora-deep"  # Kendi model adın
)

models["DIVERSE"] = PeftModel.from_pretrained(
    base_model,
    "B0DH1i/qwen-coder-lora-diverse"  # Kendi model adın
)
```

#### Adım 4: Space Ayarları
- **Hardware**: CPU (ücretsiz) veya GPU (ücretli)
- **Visibility**: Public
- **Save**

Space otomatik build olacak ve yayına girecek!

---

## 🎮 Kullanım:

### Tek Model Modu:
1. **Problem Description**: Sorunuzu yazın
2. **Select Model**: DEEP veya DIVERSE seç
3. **Settings** (opsiyonel):
   - Temperature: 0.7 (önerilen)
   - Top P: 0.95
   - Max Tokens: 512
4. **Generate Code** tıkla

### Karşılaştırma Modu:
1. **Problem Description**: Sorunuzu yazın
2. **Settings** ayarla
3. **Compare Both Models** tıkla
4. İki model çözümünü yan yana gör

---

## ⚙️ Parametre Rehberi:

### 🌡️ Temperature:
- **0.1-0.5**: Deterministik, odaklı kod
- **0.6-0.9**: Dengeli (önerilen)
- **1.0-2.0**: Yaratıcı, çeşitli çözümler

### 🎲 Top P:
- **0.9-0.95**: Kod üretimi için ideal
- Düşük: Daha odaklı
- Yüksek: Daha çeşitli

### 📏 Max Tokens:
- **128-256**: Kısa fonksiyonlar
- **512**: Orta karmaşıklık (önerilen)
- **1024**: Karmaşık implementasyonlar

### 🎰 Sampling:
- **Aktif**: Temperature ve Top P kullanır (önerilen)
- **Pasif**: Greedy decoding (deterministik)

---

## 🌐 HuggingFace Space Linki:

Training bitince model'leri yükle, sonra:

```
https://huggingface.co/spaces/B0DH1i/qwen-coder-lora-demo
```

Bu linki sunumda paylaş!

---

## 📸 Ekran Görüntüleri:

Demo çalışınca ekran görüntüleri al:
1. Tek model modu
2. Karşılaştırma modu
3. Settings paneli
4. Örnek çıktılar

Sunumda kullan!

---

## 🎯 Sunum İçin:

Demo linkini sunumda göster:
- **Canlı demo**: Space linkini paylaş
- **Video**: Ekran kaydı al
- **Ekran görüntüleri**: Önemli özellikleri göster

---

**Mükemmel bir demo olacak!** 🚀
