# Görev 2: Dataset Analizi Sonuçları

**Tarih**: 2 Aralık 2024  
**Amaç**: DEEP ve DIVERSE dataset'lerinin yapısını anlamak

## Dataset Bilgileri

### DEEP Dataset
- **HuggingFace**: Naholav/CodeGen-Deep-5K
- **Toplam Örnek**: 5,000
- **Boyut**: 55.2 MB
- **Özellik**: Daha derin reasoning trace'leri içerir

### DIVERSE Dataset
- **HuggingFace**: Naholav/CodeGen-Diverse-5K
- **Toplam Örnek**: 5,000
- **Boyut**: 56.6 MB
- **Özellik**: Daha çeşitli problem tipleri içerir

## Field Yapısı

Her iki dataset de aynı field'lara sahip:

| Field | Açıklama |
|-------|----------|
| `unique_id` | Benzersiz tanımlayıcı |
| `id` | Problem ID |
| `input` | Problem açıklaması |
| `source` | Kaynak (atcoder, codeforces, vb.) |
| `license` | Lisans bilgisi |
| `dataset` | Dataset adı |
| `split` | Train/test split |
| `output` | **Reasoning trace + kod** (`<think>` tag'leri ile) |
| `solution` | **Sadece temiz kod** |
| `difficulty` | Zorluk seviyesi (1-9) |

## Kritik Fark: output vs solution

### `output` field:
```python
<think>
This problem requires calculating the sum of the first N natural numbers.
The first child gets 1 candy, second gets 2, and so on up to the Nth child.
I can use the formula: sum = N * (N + 1) / 2
</think>

def main():
    N = int(input().strip())
    total_candies = N * (N + 1) // 2
    print(total_candies)
```

### `solution` field:
```python
def main():
    N = int(input().strip())
    total_candies = N * (N + 1) // 2
    print(total_candies)
```

## İstatistikler

### Zorluk Dağılımı

#### DEEP Dataset:
- Seviye 1: 235 (4.7%)
- Seviye 2: 392 (7.8%)
- Seviye 3: 844 (16.9%)
- Seviye 4: 248 (5.0%)
- **Seviye 5: 2,247 (44.9%)** ← En fazla
- Seviye 6: 594 (11.9%)
- Seviye 7: 390 (7.8%)
- Seviye 8: 44 (0.9%)
- Seviye 9: 6 (0.1%)

#### DIVERSE Dataset:
- Seviye 1: 211 (4.2%)
- Seviye 2: 369 (7.4%)
- Seviye 3: 789 (15.8%)
- Seviye 4: 247 (4.9%)
- **Seviye 5: 2,272 (45.4%)** ← En fazla
- Seviye 6: 599 (12.0%)
- Seviye 7: 456 (9.1%)
- Seviye 8: 45 (0.9%)
- Seviye 9: 12 (0.2%)

**Not**: Her iki dataset de orta zorlukta (seviye 5) yoğunlaşmış.

### Uzunluk İstatistikleri (İlk 1000 Örnek)

| Dataset | Solution Ortalama | Output Ortalama | Oran |
|---------|-------------------|-----------------|------|
| DEEP | 980 karakter | 5,754 karakter | ~5.9x |
| DIVERSE | 598 karakter | 4,166 karakter | ~7.0x |

**Gözlem**: 
- DEEP dataset'te solution'lar daha uzun (daha karmaşık kod)
- Output field, solution'dan ~6-7x daha uzun (reasoning trace nedeniyle)

## Kaynak Dağılımı

Problemler şu kaynaklardan alınmış:
- AtCoder
- Codeforces
- (Diğer competitive programming platformları)

## Training Stratejisi

### Bu Projede Kullanılacak:
✅ **`solution` field** - Sadece temiz kod (code-only training)

### Opsiyonel (Tavsiye Edilen):
⭐ **`output` field** - Reasoning trace ile eğitim
- Daha iyi problem anlama
- Adım adım düşünme yeteneği
- Ama daha uzun context length gerektirir (8192 token)

## Önemli Notlar

### 1. Yanlış Çözümler Hakkında
⚠️ **ÇOK ÖNEMLİ**: Dataset'te yanlış çözümler varsa **FİLTRELEMEYİN!**

**Sebep**: 
- Model yanlış çözümlerden de öğrenir
- Robustness artar
- Referans: [OpenCodeReasoning paper](https://arxiv.org/pdf/2504.01943)

### 2. Dataset Kaynağı
- Problem açıklamaları: NVIDIA OpenCodeReasoning dataset
- Reasoning ve çözümler: DeepSeek API ile üretilmiş

### 3. Train/Test Split
- Şu an sadece `train` split var
- Training sırasında manuel split yapılacak (90/10)

## Context Length Gereksinimleri

| Training Tipi | Field | Context Length |
|---------------|-------|----------------|
| Code-only | `solution` | 1024 token |
| Reasoning | `output` | 8192 token |

## Sonraki Adımlar

1. ✅ Görev 1: Base model test - Tamamlandı
2. ✅ Görev 2: Dataset analizi - Tamamlandı
3. ⏭️ **Görev 3: LoRA Fine-tuning**
   - DEEP dataset ile training
   - DIVERSE dataset ile training
4. ⏭️ Görev 4: Checkpoint değerlendirme
5. ⏭️ Görev 5: Final değerlendirme

---

**Hazır**: Dataset'ler indirildi ve yapıları anlaşıldı. Training'e başlanabilir!
