# Colab Test - Hızlı Başlangıç

## ✅ Colab'da Çalıştır:

### 1. GPU Kontrolü
```python
!nvidia-smi
```

### 2. Paketleri Kur
```python
!pip install -q torch transformers peft datasets accelerate bitsandbytes tqdm
```

### 3. Projeyi İndir
```python
!git clone https://github.com/B0DH1i/Lora-fine-tune.git
%cd Lora-fine-tune
!ls -la
```

### 4. Config Test
```python
import sys
import os

# Path ekle
sys.path.insert(0, '/content/Lora-fine-tune')
os.chdir('/content/Lora-fine-tune')

# Import test
from config.training_config import TrainingConfig
from config.model_config import ModelConfig

print("✓ Config başarıyla yüklendi!")
print(f"Learning rate: {TrainingConfig.learning_rate}")
print(f"LoRA rank: {ModelConfig.lora_r}")
```

### 5. Model Test
```python
from models.model_loader import load_model_and_tokenizer

print("Model yükleniyor...")
model, tokenizer = load_model_and_tokenizer(
    use_flash_attention=False,
    load_in_8bit=False
)
print("✓ Model yüklendi!")
```

## 🎯 Bu Çalışırsa:

Tam training notebook'unu kullanabilirsin:
- `colab_training_deep.ipynb`
- `colab_training_diverse.ipynb`

## ⚠️ Sorun Varsa:

Hangi adımda hata aldığını bana söyle!
