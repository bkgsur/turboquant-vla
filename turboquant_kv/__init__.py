"""
TurboQuant KV Cache Quantization Library

Provides efficient KV cache compression for transformer models using
residual window quantization (recent tokens FP16, historical tokens INT4/INT3).

Key modules:
- quantizer: Core KVQuantizer class for quantization/dequantization
- calibration: Calibrator for computing quantization parameters
- kernels: Optimized attention kernels with quantized KV cache
- config: Configuration dataclasses for quantization settings
"""

__version__ = "0.1.0"
__author__ = "Suresh Gopalakrishnan"

# Lazy imports (populate as modules are implemented)
# from .quantizer import KVQuantizer
# from .calibration import Calibrator
# from .config import QuantizerConfig

__all__ = [
    "KVQuantizer",
    "Calibrator",
    "QuantizerConfig",
]
