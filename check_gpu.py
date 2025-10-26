#!/usr/bin/env python3
"""
GPU and System Diagnostics for RunPod
"""
import torch
import sys
import os
import subprocess

def check_system():
    print("=" * 50)
    print("SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available - training will run on CPU!")
    
    # System memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"System RAM: {memory.total / 1024**3:.1f} GB")
        print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    except ImportError:
        print("psutil not available for memory check")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("NVIDIA-SMI OUTPUT")
            print("=" * 50)
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed")
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
    
    # Check environment variables
    print("\n" + "=" * 50)
    print("ENVIRONMENT VARIABLES")
    print("=" * 50)
    cuda_vars = [k for k in os.environ.keys() if 'CUDA' in k.upper()]
    for var in cuda_vars:
        print(f"{var}: {os.environ[var]}")

def test_model_loading():
    print("\n" + "=" * 50)
    print("MODEL LOADING TEST")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "mistralai/Mistral-7B-v0.1"
        print(f"Testing model loading: {model_name}")
        
        # Test tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading with minimal settings
        print("Loading model with 4-bit quantization...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=6,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded successfully")
        
        # Check device placement
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        
        del model
        torch.cuda.empty_cache()
        print("✅ Model unloaded successfully")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_system()
    test_model_loading()
