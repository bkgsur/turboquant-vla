# SimpleVLA: A Minimal Vision-Language-Action Model

Build a simple VLA from scratch to predict 2D coordinates from images.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic data**:
   ```bash
   python scripts/1_generate_data.py
   ```

3. **Train the model**:
   ```bash
   python scripts/2_train.py
   ```

4. **Run inference**:
   ```bash
   python scripts/3_inference.py
   ```

## Project Structure

```
simplevla/
├── data/
│   ├── raw/              # Generated images stored here
│   ├── labels.csv        # All image labels
│   ├── train.csv         # Training set (80%)
│   └── val.csv           # Validation set (20%)
├── models/
│   └── simplevla_checkpoint.pt   # Trained model (after training)
├── scripts/
│   ├── 1_generate_data.py        # Synthetic data generation
│   ├── 2_train.py                # Training loop
│   └── 3_inference.py            # Inference pipeline
├── README.md
├── requirements.txt
└── LEARNING_PATH.md      # Complete learning guide
```

## Architecture

- **Vision Encoder**: Pre-trained ViT-small (frozen)
- **Action Head**: 2-layer MLP (trainable)
- **Input**: RGB images (480x640)
- **Output**: 2D coordinates (x, y)

## Learning

See [LEARNING_PATH.md](LEARNING_PATH.md) for the complete learning guide covering all phases.

## Progress

- [ ] Phase 1: Understand the problem
- [ ] Phase 2: Design architecture
- [ ] Phase 3: Prepare data
- [ ] Phase 4: Training pipeline
- [ ] Phase 5: Inference

---

**Start with**: Read `LEARNING_PATH.md` → Implement `1_generate_data.py`
