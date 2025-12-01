"""Model and LoRA Configuration"""

class ModelConfig:
    """Model ve LoRA ayarları"""
    
    # Base Model
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # LoRA Configuration
    lora_r = 32  # Rank: {16, 32, 64} arasından seçilebilir
    lora_alpha = 64  # alpha = r * 2
    lora_dropout = 0.1
    
    # Target Modules - Hem attention hem MLP layer'lara uygula
    lora_target_modules = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
    
    # Task Type
    task_type = "CAUSAL_LM"
    
    # Quantization (OOM durumunda)
    load_in_8bit = False
    load_in_4bit = False
