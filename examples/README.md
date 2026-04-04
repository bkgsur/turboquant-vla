# Examples — Usage Demonstrations

Working examples and Jupyter notebooks demonstrating TurboQuant VLA usage.

## Notebooks

### `octo_lerobot_inference.ipynb`
**Purpose**: End-to-end inference example with Octo-small + TurboQuant

**What it covers**:
- Load Octo-small model from HuggingFace
- Initialize TurboQuant quantizer with configuration
- Calibrate on sample action trajectory
- Run inference loop (process image, predict action, measure latency)
- Visualize memory usage and latency

**Use cases**:
- Understanding the full pipeline
- Debugging quantization behavior
- Prototyping new quantization strategies

**Run**:
```bash
jupyter notebook examples/octo_lerobot_inference.ipynb
```

## Scripts

### `turbopi_demo.py`
**Purpose**: Demonstrate full pipeline on actual TurboPi hardware

**What it does**:
- Connect to TurboPi camera (or simulation)
- Load Octo-small + TurboQuant model
- Run closed-loop control loop
- Log metrics and any errors

**Use cases**:
- End-to-end validation on real robot
- Performance profiling on actual hardware
- Debugging deployment issues

**Run**:
```bash
# On development machine (simulate TurboPi)
python examples/turbopi_demo.py --simulate

# On TurboPi hardware
ssh pi@pi-4b.local
python examples/turbopi_demo.py --device /dev/video0
```

## Data Requirements

Examples assume:
- Access to HuggingFace models (Octo-small)
- Sample images for inference (or camera input on Pi)
- Optional: action history for trajectory visualization

## Extending Examples

To add your own example:

1. **Jupyter notebook**: 
   - Create `examples/YOUR_EXAMPLE.ipynb`
   - Add markdown cells explaining each step
   - Include plots/visualizations
   - Note dependencies and expected outputs

2. **Python script**:
   - Create `examples/your_example.py`
   - Use argparse for configuration
   - Log metrics to `logs/` or `results/`
   - Document usage in this README

3. **Update this README**:
   - Add section with purpose, use cases, run command
   - Keep consistent with existing examples

## Common Tasks

**Load a quantized model**:
```python
from octo_integration import OctoQuantizedInference

inference = OctoQuantizedInference(
    model_id="google/octo-base-1B",
    quantizer_config=QuantizerConfig(
        bit_width=4,
        window_size=512,
    )
)

# Run inference
action = inference.infer_action(image, history)
```

**Measure latency and memory**:
```python
import time
import torch

start = time.time()
with torch.no_grad():
    action = model(image, history)
latency = time.time() - start

memory = torch.cuda.max_memory_allocated() / 1e9  # GB
```

**Save and load quantized model**:
```python
# Save
quantizer.save("checkpoints/octo_quantized_4bit.pt")

# Load
quantizer.load("checkpoints/octo_quantized_4bit.pt")
```

## Troubleshooting

**Notebook kernel dies**:
- Memory issue: reduce batch size or model size
- GPU memory: check `nvidia-smi`
- Restart kernel and try again

**Slow inference**:
- Check device (CPU vs GPU): `torch.cuda.is_available()`
- Profile with `torch.profiler` or `py-spy`
- Check model is using quantization (verify cache size)

**Model not found**:
- Check HuggingFace token: `huggingface-cli login`
- Verify model exists: `huggingface-cli repo-info google/octo-base-1B`
- Download manually: `huggingface-cli download google/octo-base-1B`

## Dependencies

Examples require:
- `turboquant_kv` (local package)
- `torch`, `transformers` (core)
- `jupyter` (notebooks)
- Optional: `lerobot` (LeRobot integration)
- Optional: `opencv-python` (image processing)

Install with:
```bash
pip install -e .[dev]  # or uv sync
```
