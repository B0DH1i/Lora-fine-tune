"""Training Loop Setup"""

from transformers import Trainer, TrainingArguments
from config.training_config import TrainingConfig
from training.callbacks import LoggingCallback, EarlyStoppingCallback
from data.data_collator import DataCollatorForCausalLM
import os

def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    run_name
):
    """
    Trainer'ı yapılandır
    
    Args:
        model: LoRA ile yapılandırılmış model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Checkpoint kayıt dizini
        run_name: Deneme adı (logging için)
    
    Returns:
        Configured Trainer
    """
    config = TrainingConfig()
    
    # Training arguments
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        run_name=run_name,
        
        # Training
        num_train_epochs=config.max_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optim=config.optimizer_type,
        max_grad_norm=config.max_grad_norm,
        
        # Scheduler
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=config.per_device_batch_size,
        
        # Logging
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,  # Son 5 checkpoint'i sakla
        
        # Memory optimization
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,  # bfloat16 precision
        
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
        
        # Misc
        report_to="none",  # WandB kullanmak isterseniz "wandb" yapın
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Data collator
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)
    
    # Callbacks
    callbacks = [
        LoggingCallback(log_dir=os.path.join(output_dir, "logs")),
        EarlyStoppingCallback(patience=config.early_stopping_patience)
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    return trainer
