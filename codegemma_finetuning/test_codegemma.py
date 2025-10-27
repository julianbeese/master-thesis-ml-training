#!/usr/bin/env python3
"""
Test Script für CodeGemma Pipeline
Testet ob alle Abhängigkeiten und die Pipeline korrekt funktionieren
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test ob alle benötigten Pakete importiert werden können"""
    try:
        import torch
        import transformers
        import peft
        import datasets
        import pandas
        import sklearn
        import yaml
        from huggingface_hub import login
        print("✅ Alle Pakete erfolgreich importiert")
        return True
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False

def test_gpu():
    """Test GPU-Verfügbarkeit"""
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU verfügbar: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠️  Keine GPU verfügbar - Training wird auf CPU laufen")
        return False

def test_config():
    """Test ob Konfigurationsdatei korrekt geladen werden kann"""
    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "training_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Konfigurationsdatei erfolgreich geladen")
        print(f"   Model: {config['model']['base_model']}")
        print(f"   Batch Size: {config['training']['batch_size']}")
        print(f"   Learning Rate: {config['training']['learning_rate']}")
        return True
    except Exception as e:
        print(f"❌ Konfigurationsfehler: {e}")
        return False

def test_data_loader():
    """Test ob Data Loader funktioniert"""
    try:
        from data_loader import CodeGemmaDataLoader
        config_path = Path(__file__).parent / "config" / "training_config.yaml"
        data_loader = CodeGemmaDataLoader(str(config_path))
        label_info = data_loader.get_label_info()
        print("✅ Data Loader erfolgreich initialisiert")
        print(f"   Labels: {label_info['label_names']}")
        return True
    except Exception as e:
        print(f"❌ Data Loader Fehler: {e}")
        return False

def main():
    """Haupttest-Funktion"""
    print("🧪 CodeGemma Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Pakete", test_imports),
        ("GPU", test_gpu),
        ("Konfiguration", test_config),
        ("Data Loader", test_data_loader)
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
    print(f"📊 Tests: {passed}/{total} erfolgreich")
    
    if passed == total:
        print("🎉 Alle Tests erfolgreich! Pipeline ist bereit.")
        return True
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Bitte Fehler beheben.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
