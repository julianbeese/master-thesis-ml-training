#!/bin/bash
# CodeGemma Fine-Tuning Start Script

echo "🚀 Starte CodeGemma Fine-Tuning Pipeline"
echo "========================================"

# Setze Hugging Face Token (optional)
export HF_TOKEN="***REMOVED***"

# Wechsle ins src Verzeichnis
cd "$(dirname "$0")/src"

# Führe Training aus
echo "📁 Arbeitsverzeichnis: $(pwd)"
echo "🔧 Starte Training..."
python train.py

echo "✅ Training abgeschlossen!"
