"""Base Model Loading"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config.model_config import ModelConfig
from config.training_config import TrainingConfig

def load_model_and_tokenizer(use_flash_attention=True, load_in_8bit=False):
    """
    Base model ve tokenizer'ı yükle
    
    Args:
        use_flash_attention: Flash Attention 2 kullan (memory optimization)
        load_in_8bit: 8-bit quantization (OOM durumunda)
    
    Returns:
        model, tokenizer
    """
    config = ModelConfig()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Pad token ayarla
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading arguments
    model_kwargs = {
        "pretrained_model_name_or_path": config.model_name,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }
    
    # Flash Attention 2 (Windows'ta çalışmıyor, atla)
    # if use_flash_attention:
    #     model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # 8-bit quantization (OOM durumunda)
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model_kwargs["quantization_config"] = quantization_config
    
    # Model yükle
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # Gradient checkpointing
    if TrainingConfig.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, tokenizer
