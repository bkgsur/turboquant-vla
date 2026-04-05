# SimpleVLA: Learning Path to Build a VLA from Scratch

A complete guide to understanding and building the simplest Vision-Language-Action model.

---

## **Phase 1: Understand the Problem (30 mins)**

### What are you building?
A model that takes an image and predicts 2D coordinates (x, y).

```
Camera sees a scene → Model looks at image → Outputs: (x, y) position
Example: "Look at the red ball" → Model predicts (120, 340) on screen
```

### Why this is a "VLA"
- **V**ision: Processes images
- **L**anguage: (We'll skip for now, add later)
- **A**ction: Outputs an action (coordinates = where to move)

### Why this is the simplest
- Single output (2 numbers, not 7 joint angles or sequences)
- No temporal reasoning (one image → one action, not video → sequence)
- No language (pure vision)
- Synthetic data (no robot needed)

---

## **Phase 2: Design Your Model Architecture (45 mins)**

### Step 1: Choose a vision encoder
**Question**: How do we extract useful features from an image?

**Options**:
- **CNN (ResNet)**: Traditional, works, but learns from scratch (slow)
- **ViT (Vision Transformer)**: Modern, but overkill for simple task
- **Pre-trained ViT**: Best option — someone already trained on ImageNet, we reuse

**Your decision**: Use **pretrained ViT-small from Hugging Face**
- Already understands edges, objects, textures from 1M+ images
- We don't train this part, just extract features
- Output: 768-dim vector per image

### Step 2: Choose an action head
**Question**: How do we go from image features → (x, y) coordinates?

**Options**:
- MLP with 1-2 layers: Simple, works
- Complex decoder: Overkill
- Attention mechanism: Overkill

**Your decision**: Simple **2-layer MLP**
```
Input: 768-dim feature vector (from ViT)
  ↓
Hidden layer: 256 neurons + ReLU
  ↓
Output layer: 2 neurons (x, y)
```

### Complete architecture overview
```
Image (480x640x3)
  ↓
[Pre-trained ViT-small encoder] ← frozen, no training
  ↓
768-dim feature vector
  ↓
[2-layer MLP] ← this is what we TRAIN
  ↓
(x, y) prediction
```

**Key insight**: We're not building a fancy model. We're reusing a pre-trained vision model and adding a tiny decision layer on top.

---

## **Phase 3: Prepare Your Data (30 mins understanding)**

### What data do you need?
Pairs of: (image, (x, y) label)

**Example**:
```
Image 1: blue square at position (100, 200) → Label: (100, 200)
Image 2: red circle at position (320, 450) → Label: (320, 450)
... 1000 examples
```

### How to generate synthetic data
**Don't overthink this**. Create a simple generator:

1. **Generate random image**: Use PIL or numpy
   - Add random colored shapes (circles, squares) at random positions
   - Add noise/background
   - Save as PNG

2. **Record the position**: If you put a red square at pixel (x, y), store the label as (x, y)

3. **Create 1000 examples**: Loop 1000 times, save images + CSV with labels

**Example logic**:
```
for i in range(1000):
    image = blank white image (480x640)
    x, y = random integers (0-640, 0-480)
    color = random RGB
    draw circle at (x, y)
    save image as "image_0001.png"
    save label as (x, y) in CSV
```

**Why synthetic?**
- No robot needed
- Full control (you know the ground truth)
- Fast iteration
- Perfect for learning

---

## **Phase 4: Training Pipeline (45 mins understanding)**

### Step 1: Load data
**What you're doing**: 
- Read PNG images
- Load corresponding (x, y) labels from CSV
- Convert to tensors (PyTorch format)
- Create a DataLoader (batches of 32 images)

**Concepts to understand**:
- DataLoader handles batching, shuffling, parallel loading
- You need train/validation splits (80/20)

### Step 2: Forward pass
**What you're doing**:
```
1. Feed image batch to ViT encoder → get 768-dim features
2. Feed features to MLP → get (x, y) predictions
3. Compare predictions to true labels
```

### Step 3: Loss function
**What you're doing**: Measure "how wrong" your predictions are

**Best choice**: **L2 loss (Mean Squared Error)**
```
For each (x, y) prediction:
  error = (predicted_x - true_x)² + (predicted_y - true_y)²
  
Average across batch → single loss number
```

**Why L2?** Coordinates are continuous numbers, L2 penalizes far predictions heavily (good for spatial tasks)

### Step 4: Optimization
**What you're doing**: Update MLP weights to reduce loss

**Best choice**: **Adam optimizer**
```
Learn rate: 0.001 (standard starting point)
Batch size: 32 (balance memory & stability)
Epochs: 20-50 (keep going until validation loss stops improving)
```

**Concepts to understand**:
- Epoch = one pass through all training data
- Validation loss = check on held-out data (not training)
- Early stopping = stop if validation loss increases (overfitting)

### Step 5: Evaluate
**What you're measuring**:
```
1. Final validation loss (should decrease over time)
2. Test accuracy: (predicted_x - true_x) < 10 pixels? (adjust threshold)
3. Visualize: Show some predictions vs ground truth on test set
```

---

## **Phase 5: Inference (Understanding)**

Once trained, your pipeline is:

```
New image
  ↓
[Pretrained ViT] → 768-dim feature
  ↓
[Trained MLP] → (x, y) prediction
  ↓
Output: "Look at pixel (x, y)"
```

**Speed**: Should be <100ms on CPU for a single image (your VLA is fast!)

---

## **Project Structure (Create This)**

```
simplevla/
├── data/
│   ├── raw/
│   │   ├── image_0001.png
│   │   ├── image_0002.png
│   │   └── ... (1000 images)
│   ├── labels.csv  (columns: image_name, x, y)
│   ├── train.csv   (80% of labels)
│   └── val.csv     (20% of labels)
├── models/
│   └── simplevla_checkpoint.pt  (saved model after training)
├── scripts/
│   ├── 1_generate_data.py       (your synthetic data generator)
│   ├── 2_train.py               (your training loop)
│   └── 3_inference.py           (test on new images)
├── README.md
├── requirements.txt
└── LEARNING_PATH.md
```

---

## **Implementation Checklist (In This Order)**

- [ ] **Create folder structure** above
- [ ] **Write 1_generate_data.py**
  - [ ] Create random images with colored shapes
  - [ ] Record ground truth positions in CSV
  - [ ] Test: Load 1 image, visualize it with marked position
  
- [ ] **Write 2_train.py**
  - [ ] Load ViT-small from Hugging Face (pretrained)
  - [ ] Create simple 2-layer MLP on top
  - [ ] Load your synthetic data into DataLoader
  - [ ] Training loop: forward → loss → backward → optimizer step
  - [ ] Track validation loss, plot learning curves
  - [ ] Save best checkpoint
  
- [ ] **Write 3_inference.py**
  - [ ] Load saved model
  - [ ] Run on a test image
  - [ ] Visualize: draw prediction dot on image
  - [ ] Compare to ground truth

---

## **Key Concepts to Understand (As You Build)**

| Concept | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Pre-trained model** | Weights learned from ImageNet (1M images) | Gives you a "free" visual understanding |
| **Frozen encoder** | ViT weights don't change during training | Saves memory, training time, prevents overfitting |
| **Fine-tuning head** | Only MLP weights are trained | Small model = fast training |
| **Loss function** | Measures prediction error | Guides learning direction |
| **DataLoader** | Batches images for efficient GPU/CPU use | Makes training fast |
| **Validation set** | Unseen data to check overfitting | Prevents memorizing training data |
| **Learning rate** | How big are weight updates? | Too high = unstable, too low = slow |

---

## **Expected Results**

After training ~20 epochs:
- **Validation loss**: Should drop from ~50,000 → ~100 (depends on image size)
- **Accuracy**: Most predictions within 20 pixels of true position
- **Inference time**: <50ms per image on CPU
- **Model size**: ~50MB (mostly from ViT)

---

## **Debugging Tips (When Something Goes Wrong)**

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease | Learning rate too low | Increase to 0.01 |
| Loss jumps wildly | Learning rate too high | Decrease to 0.0001 |
| Training loss ↓ but val loss ↑ | Overfitting | Add dropout, reduce epochs |
| Predictions always center | Model not learning | Check data loading, verify labels aren't all same |
| Out of memory | Batch size too large | Reduce to 16 or 8 |

---

## **Next Steps (After Basic Version Works)**

1. **Add language input**: Embed instruction ("look at red"), concatenate to features
2. **Predict multiple points**: Output 10 (x,y) pairs instead of 1
3. **Temporal reasoning**: Input video (5 frames) instead of 1 image
4. **Real robot data**: Replace synthetic with actual robot video + joint angles

---

## **Resources You'll Need**

- **PyTorch**: `pip install torch`
- **Hugging Face Transformers**: `pip install transformers`
- **Pillow (PIL)**: Image generation/loading
- **NumPy, Pandas**: Data handling
- **Matplotlib**: Visualization

---

## **Getting Started**

1. Read through this entire document
2. Create the folder structure (data/, models/, scripts/)
3. Start with Phase 1 implementation: `1_generate_data.py`
4. Test data generation before moving to training
5. Ask questions as you build!
