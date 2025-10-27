# CodeGemma RunPod Setup

## 🚀 Schnellstart auf RunPod

### 1. Abhängigkeiten installieren
```bash
cd /workspace/master-thesis-ml-training/codegemma_finetuning/src
pip install datasets hf_transfer python-dotenv
```

### 2. Environment Variable setzen
```bash
# Option 1: .env Datei erstellen
echo "HF_TOKEN=dein_token_hier" > /workspace/master-thesis-ml-training/.env

# Option 2: Environment Variable setzen
export HF_TOKEN=dein_token_hier
```

### 3. Training starten
```bash
cd /workspace/master-thesis-ml-training/codegemma_finetuning/src
python train.py
```

## 🔧 Troubleshooting

### CSV-Dateien nicht gefunden
- **Problem**: `FileNotFoundError: CSV-Datei nicht gefunden`
- **Lösung**: Stelle sicher, dass die CSV-Dateien in `/workspace/master-thesis-ml-training/data/` existieren

### HF_TOKEN nicht gefunden
- **Problem**: `HF_TOKEN nicht gefunden!`
- **Lösung**: Setze die Environment Variable oder erstelle `.env` Datei

### Module nicht gefunden
- **Problem**: `ModuleNotFoundError: No module named 'datasets'`
- **Lösung**: `pip install datasets hf_transfer python-dotenv`

## 📊 Erwartete Ausgabe

```
✅ Umgebungsvariablen aus .env geladen
🚀 RTX 5090 Optimierungen aktiviert
✅ Hugging Face Token authentifiziert
Lade Daten...
Lade CSV: /workspace/master-thesis-ml-training/data/training_data_lang_70pct.csv
Train Samples: 1638
Validation Samples: 348
Test Samples: 360
```
