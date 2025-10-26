# Supervised Fine-Tuning Pipeline

Diese Pipeline führt Supervised Fine-Tuning für Frame-Classification durch. Im Gegensatz zur Classification Pipeline ist diese speziell für generatives Fine-Tuning optimiert.

## Überblick

Das Model wird mittels Supervised Fine-Tuning (SFT) trainiert, um Textabschnitte in Frame-Kategorien zu klassifizieren:
- Conflict
- Moral Value
- Economic
- Powerlessness
- None
- Human Impact

## Struktur

```
supervised_finetuning/
├── config/
│   └── training_config.yaml    # Training-Konfiguration (optimiert für SFT)
├── src/
│   ├── data_loader.py          # SFT Data Loader
│   ├── train.py                # SFT Training Script
│   └── __init__.py
├── outputs/                    # Training Outputs
└── checkpoints/                # Model Checkpoints
```

## Unterschiede zur Classification Pipeline

| Aspekt | Classification | Supervised Fine-Tuning |
|--------|---------------|------------------------|
| **Ansatz** | Multi-Class Classification Head | Generatives Fine-Tuning |
| **Learning Rate** | 2e-4 | 5e-5 (niedriger) |
| **Epochs** | 3 | 5 (mehr) |
| **Batch Size** | 2 | 4 |
| **Fokus** | Schnelle Klassifikation | Tiefes Sprachverständnis |

## Verwendung

### 1. Training starten

```bash
cd supervised_finetuning/src
python train.py
```

Das Script:
- Lädt die Daten aus `/data/training_data_export.csv` und `/data/test_data_export.csv`
- Trainiert ein Mistral-7B Model mit LoRA
- Verwendet niedrigere Learning Rate für stabileres Fine-Tuning
- Loggt Metriken zu Weights & Biases (Projekt: "supervised-finetuning-brexit-debates")
- Speichert Checkpoints in `supervised_finetuning/checkpoints/`

### 2. Konfiguration anpassen

Bearbeite `config/training_config.yaml`:

```yaml
training:
  num_epochs: 5           # Mehr Epochs für besseres Fine-Tuning
  batch_size: 4           # Größere Batches möglich
  learning_rate: 5e-5     # Niedrigere LR für Stabilität
  gradient_accumulation_steps: 8
```

## Daten

Die Pipeline nutzt dieselben CSV-Dateien wie die Classification Pipeline:
- `/data/training_data_export.csv`
- `/data/test_data_export.csv`

Format:
```csv
chunk_text,frame_name
"Text...",Conflict
"Text...","Moral Value"
```

## Training-Strategie

### Supervised Fine-Tuning Vorteile:
1. **Besseres Sprachverständnis**: Das Model lernt nicht nur zu klassifizieren, sondern versteht den Kontext besser
2. **Transferlernen**: Kann auf ähnliche Tasks übertragen werden
3. **Robustheit**: Generalisiert besser auf unseen data

### LoRA Configuration:
- **r**: 16 (Rank der Low-Rank Matrizen)
- **alpha**: 32 (Scaling Faktor)
- **dropout**: 0.05
- **target_modules**: Alle Attention und MLP Layers

## Metriken

Das Training tracked:
- **F1 Macro**: Hauptmetrik für Modelauswahl
- **F1 Micro**: Gesamtperformance
- **F1 Weighted**: Gewichtete Performance
- **Accuracy**: Korrekte Klassifikationen
- **Precision & Recall**: Detaillierte Metriken

## Hardware-Anforderungen

- **GPU**: RTX 5090 oder ähnlich (mindestens 24GB VRAM)
- **RAM**: 32GB+ empfohlen
- **Storage**: 50GB+ für Models und Checkpoints

## Weights & Biases Integration

Das Training loggt automatisch zu W&B:
- Projekt: `supervised-finetuning-brexit-debates`
- Run Name: `mistral-7b-sft-{learning_rate}`

Setup:
```bash
wandb login
```

## Tipps für bessere Ergebnisse

1. **Learning Rate**: Starte mit 5e-5, reduziere bei Instabilität
2. **Epochs**: 5-7 Epochs typisch für SFT
3. **Early Stopping**: Patience von 3 verhindert Overfitting
4. **Batch Size**: Erhöhe wenn mehr VRAM verfügbar

## Troubleshooting

### OOM (Out of Memory)
- Reduziere `batch_size` auf 2
- Erhöhe `gradient_accumulation_steps`
- Setze `max_length` auf 384

### Training instabil
- Reduziere Learning Rate auf 3e-5
- Erhöhe `warmup_steps` auf 300
- Prüfe Data Quality

### Schlechte Performance
- Erhöhe `num_epochs` auf 7
- Experimentiere mit `lora_r` (8, 16, 32)
- Prüfe Label Balance in Daten
