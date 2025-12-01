"""LoRA Configuration Setup"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config.model_config import ModelConfig

def setup_lora(model, use_8bit=False):
    """
    LoRA konfigürasyonunu modele uygula
    
    Args:
        model: Base model
        use_8bit: 8-bit training kullanılıyor mu
    
    Returns:
        LoRA ile yapılandırılmış model
    """
    config = ModelConfig()
    
    # 8-bit training için model hazırlığı
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=config.task_type
    )
    
    # LoRA'yı modele uygula
    model = get_peft_model(model, lora_config)
    
    # Trainable parameters bilgisi
    model.print_trainable_parameters()
    
    return model
