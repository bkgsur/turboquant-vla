# TurboQuant for VLA Models on Raspberry Pi 4B

## Goal

Implement TurboQuant KV cache compression to enable real-time Vision-Language-Action (VLA) inference on **HiWonder TurboPi (Raspberry Pi 4B with Quad-core ARM Cortex-A72, 4GB RAM)**.

## Why this matters

| Problem | TurboQuant solution |
|---------|-------------------|
| VLA models need 24GB+ VRAM | Compress KV cache 4x → fits in Pi 4B's 4GB |
| KV cache explodes with action history | 3-4 bit quantization with minimal accuracy loss |
| Pi 4B can't run VLAs in real-time | TurboQuant enables 5-10 Hz (100-200ms latency) |

## Hardware

- **Primary deployment**: HiWonder TurboPi (Raspberry Pi 4B, 4GB RAM, ARM Cortex-A72) — edge robotics target
- **Development**: RTX 4050 (8GB) — simulate Pi constraints, test quantization strategies
- **Target models**: Octo-small (500M), π₀ quantized (3B params) — both designed for robotics

## Tech stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Core ML | Python + PyTorch (on-device) | VLA ecosystem; TorchScript for Pi deployment |
| Quantization | TurboQuant + ONNX Runtime | 3-4 bit for ARM inference |
| VLA framework | Octo / π₀ | Robotics-focused, Pi-optimized models |
| On-robot | LeRobot API or direct inference | Direct inference preferred for Pi's constraints |

## Development roadmap

### Phase 1: Core ML

- Study TurboQuant paper (arXiv:2504.19874)
- Study existing implementations (SGLang PR #21617, PyPI turboquant)
- Build TurboQuant KV cache as standalone Python library
- Integrate with LeRobot/OpenVLA

### Phase 2: Validation on Pi 4B

- Validate action accuracy (Octo/π₀ benchmark suite)
- Test memory footprint on actual Pi 4B hardware
- Benchmark latency on ARM Cortex-A72 (target: 5-10 Hz)
- Determine optimal bit-width (3-bit vs 4-bit for real-time)
- Compare ONNX Runtime vs PyTorch inference on ARM

### Phase 3: Optimization for Pi

- Optimize KV cache quantization for ARM (consider weight quantization too)
- Profile memory usage and identify bottlenecks
- Consider streaming inference for longer histories if needed
- Test with actual TurboPi + LeRobot integration

### Phase 4: Robot deployment

- Integrate with HiWonder TurboPi software stack
- End-to-end testing on real robot (Octo policies)
- Profile power consumption and thermal behavior
- Document deployment pipeline

## Research questions

- Does KV quantization hurt action accuracy on Octo/π₀?
- What bit-width works best (3-bit vs 4-bit) on ARM Cortex-A72?
- Can we fit quantized model + KV cache in 4GB RAM with inference overhead?
- What's the achievable inference latency (target: 5-10 Hz for real-time control)?
- Should vision tokens be compressed differently from action tokens?
- How does ONNX Runtime ARM performance compare to PyTorch on Pi?

## References

- TurboQuant paper: arXiv:2504.19874
- SGLang implementation: github.com/sgl-project/sglang/pull/21617
- LeRobot: github.com/huggingface/lerobot
- OpenVLA: github.com/openvla/openvla
