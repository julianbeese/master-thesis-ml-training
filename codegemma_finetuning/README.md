# CodeGemma Fine-Tuning Pipeline für RTX 5090

Fine-tuning pipeline für das EpistemeAI-codegemma-2-9b Modell für Frame-Classification, optimiert für RTX 5090 (31.4 GB VRAM) auf RunPod.

## Modell-Informationen

- **Base Model**: [EpistemeAI/EpistemeAI-codegemma-2-9b](https://huggingface.co/EpistemeAI/EpistemeAI-codegemma-2-9b)
- **Architektur**: Gemma-2-9B (Code-spezialisiert)
- **Größe**: 9B Parameter
- **Spezialisierung**: Code-Generation und Text-Understanding
- **Quantisierung**: Full Precision (RTX 5090 optimiert)

## RTX 5090 Optimierungen

- **VRAM Nutzung**: 95% der verfügbaren 31.4 GB
- **Batch Size**: 8 (Effective: 16 mit Gradient Accumulation)
- **Max Length**: 512 Tokens
- **LoRA Rank**: 16 (höher für bessere Anpassung)
- **Precision**: BF16 (nativ unterstützt)
- **CUDA Optimierungen**: Tensor Cores, cuDNN Benchmark

## Verwendung

### RTX 5090 Test (empfohlen)
```bash
cd codegemma_finetuning
python test_rtx5090.py
```

### Training starten
```bash
cd codegemma_finetuning
./run_training.sh
```

### Manuell
```bash
cd codegemma_finetuning/src
python train.py
```

## Konfiguration

Die Konfiguration befindet sich in `config/training_config.yaml` und ist bereits für RTX 5090 optimiert:
- **Batch Size**: 8 (RTX 5090 optimiert)
- **Learning Rate**: 2e-5
- **LoRA Parameter**: r=16, alpha=32
- **Training Epochs**: 5
- **Max Length**: 512

## Performance-Erwartungen

- **Training Speed**: ~2-3x schneller als RTX 4090
- **Memory Usage**: ~20-25 GB VRAM
- **Batch Size**: Bis zu 16 effective batch size
- **Convergence**: Bessere Konvergenz durch höhere LoRA-Rank

## Ausgabe

- **Modelle**: `outputs/`
- **Checkpoints**: `checkpoints/`
- **Logs**: Konsolen-Ausgabe mit RTX 5090 Metriken
