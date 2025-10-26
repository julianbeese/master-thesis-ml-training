#!/usr/bin/env python3
"""
Debug version of training script with better GPU detection and error handling
"""
import os
import yaml
import torch
import wandb
from pathlib import Path
from typing import Dict, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import gc

from data_loader import CSVDataLoader


def check_gpu_availability():
    """Check GPU availability and print detailed info"""
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute capability: {props.major}.{props.minor}")
        
        # Check current GPU memory
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {current_device}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    else:
        print("❌ WARNING: CUDA not available - training will run on CPU!")
        print("This will be very slow and memory-intensive.")
    
    print("=" * 60)


def compute_metrics(eval_pred):
    """Metriken für Multi-Class-Klassifikation"""
    predictions, labels = eval_pred
    
    # Argmax für predicted class
    predictions = np.argmax(predictions, axis=1)
    
    # Metriken berechnen
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def setup_model_and_tokenizer(config: Dict, label_info: Dict):
    """Initialisiert Model und Tokenizer mit LoRA"""
    model_name = config['model']['base_model']
    num_labels = label_info['num_labels']
    label2id = label_info['label2id']
    id2label = label_info['id2label']
    
    print(f"Loading model: {model_name}")
    
    # Check GPU availability first
    check_gpu_availability()
    
    # Tokenizer laden
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    # Padding Token setzen falls nicht vorhanden
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token
    
    print("Loading model...")
    
    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    try:
        # Model für Multi-Class-Klassifikation laden (speichereffizient)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            problem_type="single_label_classification",
            trust_remote_code=config['model']['trust_remote_code'],
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            load_in_4bit=True if device == "cuda" else False
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print("✅ Model loaded successfully")
        
        # Set pad_token_id in model config
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        # LoRA Configuration
        if config['training']['use_lora']:
            print("Configuring LoRA...")
            
            # Prepare model for k-bit training if using quantization
            if hasattr(model, 'is_quantized') and model.is_quantized:
                print("Preparing model for k-bit training...")
                model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                r=config['training']['lora_r'],
                lora_alpha=config['training']['lora_alpha'],
                target_modules=config['training']['lora_target_modules'],
                lora_dropout=config['training']['lora_dropout'],
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # Check memory usage after model loading
        if torch.cuda.is_available():
            print(f"GPU memory after model loading:")
            print(f"  - Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            print(f"  - Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def setup_training_args(config: Dict, output_dir: Path) -> TrainingArguments:
    """Erstellt TrainingArguments aus Config"""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['evaluation']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        optim=config['training']['optimizer'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=config['evaluation']['metric'],
        greater_is_better=True,
        report_to="wandb",
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Reduce memory usage
        dataloader_num_workers=0,     # Reduce memory usage
    )


def main():
    try:
        # Config laden
        config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Configuration loaded successfully")
        print(f"Model: {config['model']['base_model']}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Max length: {config['data']['max_length']}")
        
        # Ausgabe-Verzeichnisse erstellen
        output_dir = Path(__file__).parent.parent / config['training']['output_dir']
        checkpoint_dir = Path(__file__).parent.parent / config['training']['checkpoint_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Weights & Biases initialisieren
        wandb.init(
            project="gpt-oss-20b-brexit-debates-debug",
            config=config,
            name=f"debug-{config['training']['learning_rate']}"
        )
        
        # Daten laden
        print("Loading data...")
        data_loader = CSVDataLoader(str(config_path))
        label_info = data_loader.get_label_info()
        num_labels = label_info['num_labels']
        
        print(f"Number of labels: {num_labels}")
        print(f"Label names: {label_info['label_names']}")
        
        # Model und Tokenizer initialisieren
        model, tokenizer = setup_model_and_tokenizer(config, label_info)
        
        # Datasets erstellen
        print("Creating datasets...")
        train_dataset, validation_dataset, test_dataset = data_loader.prepare_datasets(tokenizer)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(validation_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Training Arguments
        training_args = setup_training_args(config, output_dir)
        
        # Trainer initialisieren
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Training starten
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        train_result = trainer.train()
        
        # Metriken speichern
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Model speichern
        print("\nSaving final model...")
        trainer.save_model(str(checkpoint_dir / "final_model"))
        tokenizer.save_pretrained(str(checkpoint_dir / "final_model"))
        
        # Evaluation
        print("\nEvaluating model...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        # Finale Evaluation auf Test Set
        print("\nFinal evaluation on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        print(f"\nValidation metrics:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"\nTest metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        wandb.finish()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        raise


if __name__ == "__main__":
    main()
