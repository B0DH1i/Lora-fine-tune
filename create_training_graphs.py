"""
Training Loss Grafikleri Oluşturma
Her iki model için training ve validation loss grafikleri
"""

import json
import matplotlib.pyplot as plt
import glob
import os

def create_training_graphs(log_dir, model_name, color_train='blue', color_eval='red'):
    """Tek model için grafik oluştur"""
    
    log_files = glob.glob(f"{log_dir}/training_log_*.jsonl")
    
    if not log_files:
        print(f"❌ {model_name}: Log dosyası bulunamadı")
        return None
    
    log_file = log_files[0]
    
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    # Train ve eval loss'ları ayır
    train_logs = [log for log in logs if 'train_loss' in log]
    eval_logs = [log for log in logs if 'eval_loss' in log]
    
    if not train_logs:
        print(f"❌ {model_name}: Train loss bulunamadı")
        return None
    
    # Grafik oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train Loss
    steps = [log['step'] for log in train_logs]
    losses = [log['train_loss'] for log in train_logs]
    
    ax1.plot(steps, losses, color=color_train, linewidth=2.5, label='Train Loss')
    ax1.set_xlabel('Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title(f'{model_name} - Training Loss', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Eval Loss
    if eval_logs:
        eval_steps = [log['step'] for log in eval_logs]
        eval_losses = [log['eval_loss'] for log in eval_logs]
        
        ax2.plot(eval_steps, eval_losses, color=color_eval, linewidth=2.5, 
                marker='o', markersize=8, label='Validation Loss')
        ax2.set_xlabel('Step', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax2.set_title(f'{model_name} - Validation Loss', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Kaydet
    filename = f'{model_name.lower()}_training_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ {model_name} grafiği kaydedildi: {filename}")
    
    # İstatistikler
    stats = {
        'model': model_name,
        'total_steps': len(train_logs),
        'first_loss': losses[0],
        'last_loss': losses[-1],
        'improvement': ((losses[0] - losses[-1]) / losses[0] * 100)
    }
    
    if eval_logs:
        stats['best_eval_loss'] = min(eval_losses)
        stats['best_checkpoint'] = eval_logs[eval_losses.index(min(eval_losses))]['step']
    
    return stats

def create_comparison_graph(deep_log_dir, diverse_log_dir):
    """İki modeli karşılaştırmalı grafik"""
    
    # DEEP logs
    deep_log = glob.glob(f"{deep_log_dir}/training_log_*.jsonl")[0]
    with open(deep_log, 'r') as f:
        deep_logs = [json.loads(line) for line in f]
    deep_train = [log for log in deep_logs if 'train_loss' in log]
    
    # DIVERSE logs
    diverse_log = glob.glob(f"{diverse_log_dir}/training_log_*.jsonl")[0]
    with open(diverse_log, 'r') as f:
        diverse_logs = [json.loads(line) for line in f]
    diverse_train = [log for log in diverse_logs if 'train_loss' in log]
    
    # Karşılaştırmalı grafik
    plt.figure(figsize=(12, 6))
    
    # DEEP
    deep_steps = [log['step'] for log in deep_train]
    deep_losses = [log['train_loss'] for log in deep_train]
    plt.plot(deep_steps, deep_losses, 'b-', linewidth=2.5, label='DEEP Model', alpha=0.8)
    
    # DIVERSE
    diverse_steps = [log['step'] for log in diverse_train]
    diverse_losses = [log['train_loss'] for log in diverse_train]
    plt.plot(diverse_steps, diverse_losses, 'g-', linewidth=2.5, label='DIVERSE Model', alpha=0.8)
    
    plt.xlabel('Step', fontsize=13, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=13, fontweight='bold')
    plt.title('Model Comparison - Training Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Karşılaştırma grafiği kaydedildi: model_comparison.png")

# ============================================
# MAIN - Colab'da Çalıştır
# ============================================

if __name__ == "__main__":
    from google.colab import drive, files
    drive.mount('/content/drive')
    
    print("=" * 60)
    print("TRAINING LOSS GRAFİKLERİ")
    print("=" * 60)
    
    # DEEP grafiği
    print("\n1. DEEP Model Grafiği Oluşturuluyor...")
    deep_stats = create_training_graphs(
        log_dir='/content/drive/MyDrive/lora_checkpoints/deep/logs',
        model_name='DEEP',
        color_train='#2E86DE',
        color_eval='#EE5A6F'
    )
    
    if deep_stats:
        print(f"\n📊 DEEP İstatistikleri:")
        print(f"  Toplam step: {deep_stats['total_steps']}")
        print(f"  İlk loss: {deep_stats['first_loss']:.4f}")
        print(f"  Son loss: {deep_stats['last_loss']:.4f}")
        print(f"  İyileşme: {deep_stats['improvement']:.1f}%")
        if 'best_eval_loss' in deep_stats:
            print(f"  En iyi eval loss: {deep_stats['best_eval_loss']:.4f}")
            print(f"  En iyi checkpoint: step-{deep_stats['best_checkpoint']}")
    
    # DIVERSE grafiği
    print("\n2. DIVERSE Model Grafiği Oluşturuluyor...")
    diverse_stats = create_training_graphs(
        log_dir='/content/drive/MyDrive/lora_checkpoints/diverse/logs',
        model_name='DIVERSE',
        color_train='#10AC84',
        color_eval='#F79F1F'
    )
    
    if diverse_stats:
        print(f"\n📊 DIVERSE İstatistikleri:")
        print(f"  Toplam step: {diverse_stats['total_steps']}")
        print(f"  İlk loss: {diverse_stats['first_loss']:.4f}")
        print(f"  Son loss: {diverse_stats['last_loss']:.4f}")
        print(f"  İyileşme: {diverse_stats['improvement']:.1f}%")
        if 'best_eval_loss' in diverse_stats:
            print(f"  En iyi eval loss: {diverse_stats['best_eval_loss']:.4f}")
            print(f"  En iyi checkpoint: step-{diverse_stats['best_checkpoint']}")
    
    # Karşılaştırma grafiği
    if deep_stats and diverse_stats:
        print("\n3. Karşılaştırma Grafiği Oluşturuluyor...")
        create_comparison_graph(
            deep_log_dir='/content/drive/MyDrive/lora_checkpoints/deep/logs',
            diverse_log_dir='/content/drive/MyDrive/lora_checkpoints/diverse/logs'
        )
    
    print("\n" + "=" * 60)
    print("✓ TÜM GRAFİKLER OLUŞTURULDU!")
    print("=" * 60)
    
    # Grafikleri indir
    print("\nGrafikleri indiriyor...")
    files.download('deep_training_curves.png')
    files.download('diverse_training_curves.png')
    files.download('model_comparison.png')
    
    print("\n✓ Grafikleri sunumda kullanabilirsin!")
