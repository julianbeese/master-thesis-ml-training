# Master Thesis ML Training - Projekt Übersicht

Dieses Repository enthält zwei separate Training-Pipelines für Frame-Detection in Brexit-Debatten.

## 📁 Projektstruktur

```
master-thesis-ml-training/
├── data/                           # Gemeinsame Daten
│   ├── training_data_export.csv
│   ├── test_data_export.csv
│   └── debates_brexit_chunked.duckdb
│
├── classification/                 # Pipeline 1: Classification
│   ├── config/
│   │   └── training_config.yaml
│   ├── src/
│   │   ├── train.py
│   │   ├── data_loader.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── scripts/
│   ├── outputs/
│   └── README.md
│
├── supervised_finetuning/         # Pipeline 2: Supervised Fine-Tuning
│   ├── config/
│   │   └── training_config.yaml
│   ├── src/
│   │   ├── train.py
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── outputs/
│   └── README.md
│
├── requirements.txt
├── .env
└── PROJECT_OVERVIEW.md            # Diese Datei
```

## 🎯 Pipelines im Vergleich

### Pipeline 1: Classification

**Zweck**: Schnelle Multi-Class Classification für Frame-Detection

**Eigenschaften**:
- Multi-Class Classification Head
- Schnelles Training (3 Epochs)
- Höhere Learning Rate (2e-4)
- Fokus auf Klassifikationsgenauigkeit

**Verwendung**:
```bash
cd classification/src
python train.py
```

[→ Mehr Details in classification/README.md](classification/README.md)

### Pipeline 2: Supervised Fine-Tuning

**Zweck**: Tiefes Sprachverständnis durch generatives Fine-Tuning

**Eigenschaften**:
- Supervised Fine-Tuning Ansatz
- Längeres Training (5 Epochs)
- Niedrigere Learning Rate (5e-5)
- Bessere Generalisierung

**Verwendung**:
```bash
cd supervised_finetuning/src
python train.py
```

[→ Mehr Details in supervised_finetuning/README.md](supervised_finetuning/README.md)

## 📊 Welche Pipeline wählen?

| Kriterium | Classification | Supervised Fine-Tuning |
|-----------|---------------|------------------------|
| **Training Zeit** | ⚡ Schneller (3 epochs) | 🕐 Länger (5 epochs) |
| **Genauigkeit** | ✅ Gut | ✅✅ Besser |
| **Generalisierung** | ✅ Akzeptabel | ✅✅ Sehr gut |
| **Ressourcen** | 💻 Weniger | 💻💻 Mehr |
| **Use Case** | Schnelle Prototypen | Produktions-Models |

## 🚀 Quick Start

### 1. Repository klonen und Setup

```bash
git clone <repo-url>
cd master-thesis-ml-training
pip install -r requirements.txt
```

### 2. Daten vorbereiten

Die CSV-Dateien sollten bereits im `/data` Ordner vorhanden sein:
- `training_data_export.csv`
- `test_data_export.csv`

Falls nicht, generiere sie:
```bash
cd classification/scripts
python export_duckdb_to_csv.py
```

### 3. Wähle eine Pipeline und starte Training

**Classification:**
```bash
cd classification/src
python train.py
```

**Supervised Fine-Tuning:**
```bash
cd supervised_finetuning/src
python train.py
```

## 🔧 Konfiguration

Beide Pipelines haben separate Konfigurationsdateien:
- `classification/config/training_config.yaml`
- `supervised_finetuning/config/training_config.yaml`

Wichtige Parameter:
- `batch_size`: Anpassen an verfügbare GPU
- `learning_rate`: Experimentieren für beste Ergebnisse
- `num_epochs`: Mehr Epochs = bessere Performance (aber Overfitting-Risiko)
- `lora_r`: LoRA Rank (8, 16, 32)

## 📈 Metriken und Logging

Beide Pipelines nutzen Weights & Biases für Tracking:
- Classification: `gpt-oss-20b-brexit-debates`
- Supervised Fine-Tuning: `supervised-finetuning-brexit-debates`

Setup W&B:
```bash
wandb login
```

## 🎓 Frame-Kategorien

Beide Pipelines klassifizieren in 6 Kategorien:
1. **Conflict**: Konflikte und Auseinandersetzungen
2. **Moral Value**: Moralische und ethische Werte
3. **Economic**: Wirtschaftliche Aspekte
4. **Powerlessness**: Machtlosigkeit und Hilflosigkeit
5. **Human Impact**: Auswirkungen auf Menschen
6. **None**: Keine spezifische Frame

## 💾 Datenformat

CSV-Format für beide Pipelines:
```csv
chunk_text,frame_name
"We know, of course, that...",Conflict
"I certainly would...",Moral Value
```

Spalten:
- `chunk_text`: Der zu klassifizierende Text
- `frame_name`: Die Frame-Kategorie

## 🔍 Hardware-Anforderungen

**Minimum**:
- GPU: 24GB VRAM (z.B. RTX 3090, RTX 4090)
- RAM: 32GB
- Storage: 50GB

**Empfohlen**:
- GPU: RTX 5090 (48GB VRAM)
- RAM: 64GB
- Storage: 100GB+ (für Checkpoints)

## 🐛 Troubleshooting

### OOM (Out of Memory)
```yaml
# In training_config.yaml reduzieren:
batch_size: 1
gradient_accumulation_steps: 32
max_length: 384
```

### Training zu langsam
- Erhöhe `batch_size` wenn möglich
- Reduziere `logging_steps` und `eval_steps`
- Nutze `fp16` statt `bf16` (falls GPU unterstützt)

### Schlechte Performance
- Prüfe Data Quality und Label Balance
- Experimentiere mit Learning Rate
- Erhöhe `num_epochs`
- Versuche unterschiedliche `lora_r` Werte

## 📚 Weitere Ressourcen

- [Transformers Dokumentation](https://huggingface.co/docs/transformers)
- [PEFT/LoRA Guide](https://huggingface.co/docs/peft)
- [Weights & Biases Docs](https://docs.wandb.ai)

## 🤝 Contributing

Beide Pipelines können unabhängig voneinander entwickelt werden:
1. Wähle die relevante Pipeline
2. Erstelle einen Branch
3. Teste deine Änderungen
4. Committe nur die betroffene Pipeline

## 📝 Lizenz

[Füge Lizenzinformationen hinzu]

## ✉️ Kontakt

[Füge Kontaktinformationen hinzu]
