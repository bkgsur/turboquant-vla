# TurboQuant KV Cache Quantization Library

Core library for KV cache compression in transformer models.

## Purpose

Reduces KV cache memory footprint by 3-4x through residual window quantization:
- **Recent tokens** (last N): Keep in FP16 (full precision)
- **Historical tokens**: Quantize to INT4/INT3 (4x smaller)

## Module Structure

- **`quantizer.py`** — `KVQuantizer` class for quantization/dequantization operations
- **`calibration.py`** — `Calibrator` class for computing quantization parameters from data
- **`config.py`** — `QuantizerConfig` dataclass for configuration
- **`utils.py`** — Helper functions (quantization formulas, conversions)
- **`kernels/`** — Optimized attention kernels
  - `quantized_attention.py` — `QuantizedAttention` module

## Quick Start

```python
from turboquant_kv import KVQuantizer, QuantizerConfig

# Create quantizer with config
config = QuantizerConfig(
    bit_width=4,
    per_channel=True,
    window_size=512,
    symmetric=True,
)
quantizer = KVQuantizer(config)

# Calibrate on sample data
calibration_data = [...]  # List of (k, v) tensors
quantizer.calibrate(calibration_data)

# Quantize KV cache
k_quantized, v_quantized, metadata = quantizer.quantize_kv(k, v)

# Dequantize for attention
k_dequantized, v_dequantized = quantizer.dequantize_kv(
    k_quantized, v_quantized, metadata
)
```

## Performance Targets

- **Memory reduction**: 3-4x compression on KV cache
- **Accuracy loss**: < 5% on action prediction (Octo-small)
- **Latency overhead**: < 5% (dequantization + attention still faster than FP16)

## Testing

Run tests with:
```bash
pytest turboquant_kv/tests/ -v --cov=turboquant_kv
```

## Design Decisions

- **Residual window**: Keep recent tokens FP16 to preserve temporal context
- **Per-channel quantization**: Better accuracy than per-token (more fine-grained)
- **INT4**: Sweet spot between compression (4x) and accuracy (< 5% loss)
- **Symmetric quantization**: Simpler, zero_point = 0

## Known Limitations

- Currently supports PyTorch tensors (ONNX export in Phase 3)
- Assumes attention is standard scaled-dot-product (may not work with custom attention)
- Requires calibration data (100-200 examples recommended)
