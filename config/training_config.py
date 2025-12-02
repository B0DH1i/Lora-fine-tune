
class TrainingConfig:
    
    # Learning Rate
    learning_rate = 2e-4  # Muhafazakar başlangıç
    
    # Training
    max_epochs = 3
    effective_batch_size = 16
    per_device_batch_size = 1
    gradient_accumulation_steps = 16  # effective_batch_size / per_device_batch_size
    
    # Context Length
    max_length_solution = 1024  # Sadece solution field için
    max_length_reasoning = 8192  # Reasoning ile birlikte için
    
    # Optimizer
    optimizer_type = "adamw_torch"
    weight_decay = 0.01
    max_grad_norm = 1.0  # Gradient clipping
    warmup_ratio = 0.03
    
    # Scheduler
    lr_scheduler_type = "cosine"
    
    # Early Stopping
    early_stopping_patience = 2
    
    # Logging
    logging_steps = 20
    eval_steps = 100
    save_steps = 100
    
    # Memory Optimization
    use_flash_attention_2 = False  # Windows'ta çalışmıyor
    gradient_checkpointing = True
    use_8bit = False  # Son çare - kaliteyi düşürür
    
    # System Prompts
    SYSTEM_PROMPT_SOLUTION = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."
    SYSTEM_PROMPT_REASONING = "You are an expert programmer. Use <think> tags for reasoning before writing code."
    
    # Random Seed
    seed = 42
