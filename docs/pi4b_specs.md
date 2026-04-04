# Raspberry Pi 4B Specifications & Constraints

## This Project's Target Hardware

**Device**: HiWonder TurboPi  
**SBC**: Raspberry Pi Model 4B  
**Status**: THE PRIMARY & ONLY DEPLOYMENT TARGET

---

## Hardware Specifications

### Processor
- **SoC**: Broadcom BCM2711
- **CPU**: Quad-core ARM Cortex-A72
- **Frequency**: 1.5 GHz base (up to 2.0 GHz turbo boost)
- **Architecture**: ARMv8 (64-bit)
- **Cache**: L1: 32KB, L2: 1MB per core

### Memory
- **Total RAM**: 4GB LPDDR4 (SDRAM)
- **System Reserved**: ~500MB (kernel, OS services)
- **Available for ML**: ~3.5 GB (practical)
- **For inference headroom**: ~3.0 GB (conservative)

### Storage
- **Onboard**: MicroSD card (typically 32GB+)
- **Speed**: Class 10 recommended (avoid Class 4)

### Connectivity
- **Ethernet**: Gigabit (1000 Mbps)
- **WiFi**: 802.11ac (dual-band)
- **Bluetooth**: 5.0

### Power
- **Input**: USB-C (5V, 3A recommended)
- **Typical consumption**: 3-5W idle, 10-15W under load
- **Peak**: Up to 25W (with overclocking + heavy load)

### Thermal
- **Temperature limits**: 
  - Thermal throttling starts: 80°C
  - Thermal shutdown: 85°C
- **Cooling**: Passive (aluminum heatsink) or active (fan)
- **Typical temp under load**: 60-75°C (with heatsink)

### GPIO & Connectors
- **40-pin GPIO header** (for robot control)
- **USB 3.0**: 2x ports
- **USB 2.0**: 2x micro headers
- **HDMI 2.0**: 2x micro connectors
- **Camera connector**: CSI-2 (for TurboPi camera)
- **Audio**: 3.5mm jack

---

## Performance Baseline

### CPU Performance (Single Core)
- **Integer ops**: ~1000-1200 MIPS
- **Floating-point**: ~500-600 MFLOPS (single precision)
- **Compared to**: ~50x slower than RTX 4050, ~100x slower than modern laptop CPU

### Memory Bandwidth
- **LPDDR4**: ~25 GB/s peak
- **Actual (measured)**: ~10-12 GB/s sustained
- **Important**: Memory is bottleneck on Pi, not CPU

### Power Efficiency
- **Performance per Watt**: ~200 MIPS/W
- **Advantage**: Very efficient for robotics (battery-friendly)

---

## Constraints for VLA Inference

### Memory Constraint (Critical!)
```
Total RAM:           4 GB (4096 MB)
├── OS + services:   500 MB
├── Python runtime:  100 MB
├── Model weights:   2000 MB (Octo-small FP16)
├── Activations:     400 MB (peak during inference)
├── KV cache:        ??? MB (THIS IS THE PROBLEM)
└── Reserve:         ~100 MB
```

**Without TurboQuant**: KV cache grows to 37+ MB → 3500+ MB total → **OOM**

**With TurboQuant**: KV cache fixed at 12-15 MB → 3100-3200 MB total → **FITS!**

### Latency Constraint (Real-Time)
- **Required**: 5-10 Hz (100-200 ms per inference)
- **Octo-small on Pi without quantization**: 300-400 ms (too slow, under-utilizes CPU)
- **Octo-small + TurboQuant on Pi**: 150-250 ms (achievable, good utilization)

### Thermal Constraint
- **Sustained inference**: Can run 24/7 if properly cooled
- **Throttling**: Kicks in at 80°C, reduces clock to 1.0 GHz
- **Solution**: Active cooling (fan) or thermal paste + heatsink

### Power Constraint
- **Typical inference power**: 5-10W (including inference + servos)
- **Battery operation**: Important for mobile robotics
- **Advantage of ARM**: Far more efficient than GPU

---

## Optimization Implications for This Project

### 1. Memory is King
- Every MB counts
- TurboQuant KV cache compression (3-4x) is **critical**
- No room for large intermediate buffers
- Can't afford to batch process

### 2. CPU-Bound Inference
- No GPU → all ops on ARM Cortex-A72
- Quantization (INT4) is **faster** than FP16 (less data to move)
- Dequantization overhead < benefit (fewer bytes in memory)

### 3. Thermal Awareness
- Sustained high inference load → need cooling
- Monitor CPU temp: `vcgencmd measure_temp`
- May need to throttle inference frequency if overheating

### 4. ARM-Specific Optimization
- **NEON SIMD**: ARM Cortex-A72 has NEON (SIMD) support
- **NumPy/PyTorch**: Already optimized for ARM NEON
- **ONNX Runtime**: Has ARM acceleration, use it!

### 5. Network Efficiency
- Fast ethernet available for model updates/logging
- WiFi for remote monitoring (slower, less reliable)

---

## Expected Performance Metrics

### Octo-Small + TurboQuant on Pi 4B

| Metric | Value | Notes |
|--------|-------|-------|
| **Model size** | 2 GB | FP16 weights |
| **KV cache (seq=1000)** | 12-15 MB | 3-4x compressed |
| **Total memory** | ~3.1 GB | Fits in 4GB with margin |
| **Inference latency** | 150-250 ms | Single image + history |
| **Throughput** | 4-6.7 Hz | 1 action per 150-250 ms |
| **CPU usage** | ~70-90% | Single core saturated |
| **Thermal (cooled)** | 65-75°C | Idle heatsink + 5°C rise under load |
| **Power (inference)** | 8-12 W | Typical sustained |

---

## Development vs Deployment

### Development Machine (RTX 4050)
- **Purpose**: Simulate Pi constraints, prototype faster
- **Not an end target** — just for rapid iteration
- **Memory**: Simulate by limiting to 3GB allocation
- **Latency**: Will be faster, but relative improvement transfers

### Deployment Machine (Pi 4B)
- **Purpose**: Real-world testing and deployment
- **The target** — all benchmarks must be validated here
- **Memory**: Actual 4GB constraint
- **Latency**: Real inference speed on ARM

**Rule**: Always validate on actual Pi 4B before considering done.

---

## Troubleshooting on Pi 4B

### Memory Issues
```bash
# Check available RAM
free -h

# Monitor memory during inference
watch -n 0.5 free -h
```

**If OOM**: Model doesn't fit. Need more aggressive quantization (3-bit vs 4-bit).

### Thermal Throttling
```bash
# Check CPU temperature
vcgencmd measure_temp

# Check throttling status (0x0 = no throttle)
vcgencmd get_throttled
```

**If throttling**: Add active cooling or reduce inference batch frequency.

### Slow Inference
```bash
# Check CPU frequency
vcgencmd measure_clock arm

# Check if thermal throttled to 1.0 GHz
vcgencmd get_throttled
```

**If slow**: Profile with `py-spy`, identify bottleneck, optimize that path.

### SD Card Issues
- **Slow reads**: Inference hangs waiting for disk I/O
- **Corrupted**: Unlikely but can cause mysterious failures
- **Solution**: Use Class 10 card, full NOOBS install, validate SHA256

---

## Comparison: Pi 4B vs Other Targets

| Spec | Pi 4B | RTX 4050 | Jetson Orin |
|------|-------|----------|------------|
| **Cost** | $75 | ~$300 | ~$800 |
| **Memory** | 4 GB | 8 GB | 12 GB |
| **CPU cores** | 4 ARM | 16+ x86-64 | 12 ARM |
| **GPU** | None | Yes (1280 CUDA) | Yes (12GB) |
| **TDP** | 15W | 90W | 70W |
| **Real-time capable** | ✅ TurboQuant | ✅ Baseline | ✅ Baseline |
| **Deployable** | ✅ Yes | ❌ No | ⚠️ Expensive |
| **Battery-friendly** | ✅ Yes | ❌ No | ⚠️ Medium |

**This project targets Pi 4B because**:
- Cheap (mass deployment possible)
- Efficient (mobile robotics friendly)
- Challenging (TurboQuant is necessary)

---

## Resources

- **Official**: [raspberrypi.com/products/raspberry-pi-4-model-b/](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
- **Specs**: [Broadcom BCM2711 datasheet](https://www.raspberrypi.com/documentation/computers/processors.html)
- **Cooling**: [Recommended heatsinks and fans](https://www.raspberrypi.com/products/raspberry-pi-4-case/)
- **OS**: [Raspberry Pi OS (64-bit)](https://www.raspberrypi.com/software/)

---

## Key Takeaway

**Raspberry Pi 4B is resource-constrained by design**, making TurboQuant compression **critical for success**. This is NOT a limitation — it's what makes this project interesting and valuable. Shipping real-time VLAs on a $75 board changes robotics.
