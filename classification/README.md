# Classification Pipeline

Diese Pipeline trainiert ein Multi-Class Classification Model für Frame-Detection in Brexit-Debatten.

## Überblick

Das Model klassifiziert Textabschnitte in verschiedene Frame-Kategorien:
- Conflict
- Moral Value
- Economic
- Powerlessness
- None
- Human Impact

## Struktur

```
classification/
├── config/
│   └── training_config.yaml    # Training-Konfiguration
├── src/
│   ├── data_loader.py          # Data Loading und Preprocessing
│   ├── train.py                # Training Script
│   ├── evaluate.py             # Evaluation Script
│   └── inference.py            # Inference Script
├── scripts/
│   ├── generate_training_data.py
│   ├── export_duckdb_to_csv.py
│   └── fine_tuning_pipeline.py
└── outputs/                    # Training Outputs
    └── checkpoints/            # Model Checkpoints
```

## Verwendung

### 1. Training starten

```bash
cd classification/src
python train.py
```

Das Script:
- Lädt die Daten aus `/data/training_data_export.csv` und `/data/test_data_export.csv`
- Trainiert ein Mistral-7B Model mit LoRA
- Loggt Metriken zu Weights & Biases
- Speichert Checkpoints in `classification/checkpoints/`

### 2. Evaluation

```bash
cd classification/src
python evaluate.py
```

### 3. Inference

```bash
cd classification/src
python inference.py --text "Your text here"
```

## Konfiguration

Alle Training-Parameter können in `config/training_config.yaml` angepasst werden:

- **Model**: Mistral-7B-v0.1
- **Batch Size**: 2 (mit Gradient Accumulation 16 = effektiv 32)
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **LoRA**: r=16, alpha=32

## Daten

Die Pipeline nutzt CSV-Dateien aus dem `/data` Ordner:
- `training_data_export.csv`: Training-Daten
- `test_data_export.csv`: Test-Daten

Format:
```csv
chunk_text,frame_name
"Text...",Conflict
"Text...","Moral Value"
```

## Anforderungen

- Python 3.8+
- PyTorch
- Transformers
- PEFT (für LoRA)
- Weights & Biases (optional)
- GPU mit mindestens 24GB VRAM (empfohlen: RTX 5090)
