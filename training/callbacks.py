"""Training Callbacks for Logging and Early Stopping"""

from transformers import TrainerCallback, TrainerState, TrainerControl
import json
import os
from datetime import datetime

class LoggingCallback(TrainerCallback):
    """Training ve validation loss'ları kaydet"""
    
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Log dosyası
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.jsonl")
        
        self.train_losses = []
        self.eval_losses = []
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Her log adımında çağrılır"""
        if logs is None:
            return
        
        log_entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat()
        }
        
        # Train loss
        if "loss" in logs:
            log_entry["train_loss"] = logs["loss"]
            self.train_losses.append({
                "step": state.global_step,
                "loss": logs["loss"]
            })
        
        # Eval loss
        if "eval_loss" in logs:
            log_entry["eval_loss"] = logs["eval_loss"]
            self.eval_losses.append({
                "step": state.global_step,
                "loss": logs["eval_loss"]
            })
        
        # Learning rate
        if "learning_rate" in logs:
            log_entry["learning_rate"] = logs["learning_rate"]
        
        # Dosyaya yaz
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Console'a da yazdır
        print(f"Step {state.global_step}: {log_entry}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Training bittiğinde özet kaydet"""
        summary = {
            "total_steps": state.global_step,
            "total_epochs": state.epoch,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses
        }
        
        summary_file = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining tamamlandı! Log dosyası: {self.log_file}")


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping - validation loss artarsa dur"""
    
    def __init__(self, patience=2):
        self.patience = patience
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Her evaluation'da çağrılır"""
        if metrics is None:
            return
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        
        # İyileşme var mı?
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            print(f"✓ Yeni en iyi eval loss: {eval_loss:.4f}")
        else:
            self.patience_counter += 1
            print(f"✗ Eval loss iyileşmedi ({self.patience_counter}/{self.patience})")
            
            # Patience doldu mu?
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping! {self.patience} evaluation boyunca iyileşme yok.")
                control.should_training_stop = True
        
        return control
