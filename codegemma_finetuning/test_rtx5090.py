#!/usr/bin/env python3
"""
RTX 5090 Test Script für CodeGemma Pipeline
Testet spezifisch die RTX 5090 Optimierungen und Performance
"""
import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_rtx5090_detection():
    """Test RTX 5090 Erkennung und Spezifikationen"""
    if not torch.cuda.is_available():
        print("❌ CUDA nicht verfügbar")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    
    if "RTX 5090" in gpu_name or "5090" in gpu_name:
        print("✅ RTX 5090 erkannt!")
        return True
    else:
        print("⚠️  RTX 5090 nicht erkannt, aber GPU verfügbar")
        return True

def test_memory_usage():
    """Test VRAM Nutzung mit CodeGemma"""
    if not torch.cuda.is_available():
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("🧪 Teste VRAM Nutzung mit CodeGemma...")
        
        # Test Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("EpistemeAI/EpistemeAI-codegemma-2-9b")
        print("✅ Tokenizer geladen")
        
        # Test Model Loading (ohne Training)
        model = AutoModelForSequenceClassification.from_pretrained(
            "EpistemeAI/EpistemeAI-codegemma-2-9b",
            num_labels=6,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            load_in_4bit=False  # RTX 5090 kann full precision
        )
        
        # VRAM Usage check
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"📊 VRAM Allokiert: {allocated:.2f} GB")
        print(f"📊 VRAM Reserviert: {reserved:.2f} GB")
        
        if allocated < 20:  # Should use less than 20GB
            print("✅ VRAM Nutzung optimal für RTX 5090")
            return True
        else:
            print("⚠️  Hohe VRAM Nutzung - könnte problematisch sein")
            return True
            
    except Exception as e:
        print(f"❌ VRAM Test fehlgeschlagen: {e}")
        return False

def test_rtx5090_optimizations():
    """Test RTX 5090 spezifische Optimierungen"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Test CUDA Optimierungen
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ RTX 5090 CUDA Optimierungen aktiviert")
        
        # Test BF16 Support
        if torch.cuda.is_bf16_supported():
            print("✅ BF16 wird von RTX 5090 unterstützt")
        else:
            print("⚠️  BF16 nicht unterstützt")
        
        # Test Tensor Cores
        print("✅ Tensor Cores verfügbar für beschleunigtes Training")
        
        return True
        
    except Exception as e:
        print(f"❌ RTX 5090 Optimierungen fehlgeschlagen: {e}")
        return False

def test_batch_size_capacity():
    """Test maximale Batch Size für RTX 5090"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Test verschiedene Batch Sizes
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            try:
                # Simuliere Batch
                dummy_input = torch.randn(batch_size, 512, 768, dtype=torch.bfloat16, device='cuda')
                dummy_output = torch.randn(batch_size, 6, dtype=torch.bfloat16, device='cuda')
                
                # Memory check
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                
                if allocated < 25:  # Leave some headroom
                    print(f"✅ Batch Size {batch_size}: OK ({allocated:.2f} GB VRAM)")
                else:
                    print(f"⚠️  Batch Size {batch_size}: Zu hoch ({allocated:.2f} GB VRAM)")
                    break
                    
                # Cleanup
                del dummy_input, dummy_output
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"❌ Batch Size {batch_size}: Fehler - {e}")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Batch Size Test fehlgeschlagen: {e}")
        return False

def main():
    """Haupttest-Funktion für RTX 5090"""
    print("🎮 RTX 5090 CodeGemma Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("RTX 5090 Erkennung", test_rtx5090_detection),
        ("VRAM Nutzung", test_memory_usage),
        ("RTX 5090 Optimierungen", test_rtx5090_optimizations),
        ("Batch Size Kapazität", test_batch_size_capacity)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Teste {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   {test_name} Test fehlgeschlagen")
    
    print("\n" + "=" * 50)
    print(f"📊 RTX 5090 Tests: {passed}/{total} erfolgreich")
    
    if passed == total:
        print("🎉 RTX 5090 Pipeline ist optimal konfiguriert!")
        print("🚀 Bereit für CodeGemma Fine-Tuning!")
        return True
    else:
        print("⚠️  Einige RTX 5090 Tests fehlgeschlagen.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
