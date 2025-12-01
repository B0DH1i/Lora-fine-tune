 Base Model Test Sonuçları



## Model Bilgileri

- **Model Adı**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Parametre Sayısı**: 1.5B
- **Model Boyutu**: ~3.09GB
- **Precision**: bfloat16
- **Attention**: Standart (Flash Attention 2 Windows'ta çalışmadı)

## Test Problemleri ve Sonuçlar

### Test 1: Fibonacci Sayısı

**Problem**: Write a Python function that returns the nth Fibonacci number.

**Model Çıktısı**:
```python
def fibonacci(n):
    if n <= 0:
        return "Input should be a positive integer"
    elif n == 1:
        return 0
    elif n == 2:
        return 1

    a, b = 0, 1
    for i in range(2, n+1):
        a, b = b, a+b

    return b
```

**Değerlendirme**: ✅ Doğru
- İteratif yaklaşım kullanmış
- Edge case'leri kontrol ediyor
- Temiz ve okunabilir kod

---

### Test 2: Palindrome Kontrolü

**Problem**: Write a Python function that checks if a string is a palindrome.

**Model Çıktısı**:
```python
def is_palindrome(text):
    # Remove non-alphabetic characters and convert to lowercase
    cleaned_text = ''.join(char.lower() for char in text if char.isalpha())

    # Check if the cleaned text is equal to its reverse
    return cleaned_text == cleaned_text[::-1]
```

**Değerlendirme**: ✅ Doğru (kısmi)
- Alfabetik karakterleri temizliyor
- Case-insensitive kontrol yapıyor
- Kod tamamlanmamış gibi görünüyor ama mantık doğru

---

### Test 3: Two Sum

**Problem**: Given an array of integers nums and an integer target, return indices of the two numbers that add up to target.

**Model Çıktısı**:
```python
def twoSum(nums, target):
    # Create a dictionary to store numbers and their indices
    num_to_index = {}

    # Iterate over the list of numbers
    for i, num in enumerate(nums):
        # Calculate the complement needed to reach the target
        complement = target - num

        # Check if the complement is already in the dictionary
        if complement in num_to_index:
            # Return the indices of the two numbers that add up to the target
            return [num_to_index[complement], i]

        # Add the current number and its index to the dictionary
        num_to_index[num] = i

    # If no solution is found, raise an exception or return None
    raise ValueError("No two sum solution")
```

**Değerlendirme**: ✅ Doğru
- Hash map yaklaşımı (O(n) complexity)
- Optimal çözüm
- İyi açıklanmış

---
