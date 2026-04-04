# Scripts — Benchmarks, Profiling, and Deployment

Standalone scripts for testing, benchmarking, and profiling TurboQuant VLA.

## Phase-Specific Scripts

### Phase 1: Core Library Validation

- **`test_turboquant_existing.py`** — Test existing TurboQuant implementations (SGLang, PyPI package)
  - Load Octo-small, apply quantization, measure memory savings
  - Decide: build custom vs use existing

### Phase 2: Model Integration & Pi Validation

- **`benchmark_octo_quantized.py`** — Benchmark Octo-small on RTX 4050
  - Measure latency (per-action inference time)
  - Measure memory (peak GPU usage)
  - Measure accuracy (action MSE vs baseline)
  - Output: `docs/octo_benchmark_baseline.md`

- **`benchmark_octo_pi.py`** — Benchmark Octo-small on Raspberry Pi 4B
  - Load model, run 100 inferences
  - Measure per-action latency (target: 100-200ms for 5-10 Hz)
  - Measure peak memory (target: < 3GB total)
  - Measure CPU temp (check for thermal throttling)
  - Output: `docs/pi_benchmark_results.md`

- **`test_accuracy.py`** — Accuracy validation
  - Run Octo-small on evaluation set
  - Compare quantized vs baseline
  - Action prediction MSE
  - Output: `docs/accuracy_validation.md`

### Phase 3: ONNX Export & Optimization

- **`benchmark_octo_pi_optimized.py`** — Benchmark optimized ONNX model on Pi
  - Load ONNX model with ONNX Runtime
  - Measure latency vs PyTorch baseline
  - Identify remaining bottlenecks
  - Output: `docs/pi_profiling_report.md`

- **`profile_pi_inference.py`** — Detailed profiling on Pi
  - Layer-by-layer latency breakdown
  - Memory timeline during inference
  - CPU usage per operation
  - Output: `docs/pi_profiling_detailed.md`

### Phase 4: Robot Deployment

- **`turbopi_demo.py`** — End-to-end demo on TurboPi
  - Load Octo-small + TurboQuant on Pi
  - Process camera frames
  - Execute actions on robot
  - Log metrics and any errors
  - Output: `logs/turbopi_demo_TIMESTAMP.log`

### Phase 5: Documentation

- **`run_all_benchmarks.sh`** — Run full benchmark suite
  - Orchestrate all benchmarks in order
  - Collect results into single report
  - Generate comparison tables

## Running Scripts

### On Development Machine (RTX 4050)
```bash
# Activate uv environment
source turboquant-vla-dev/bin/activate
# Or: uv sync

# Run benchmark
python scripts/benchmark_octo_quantized.py --config config/octo-small.json

# Run with profiling
python scripts/benchmark_octo_quantized.py --profile --profile-output profiles/
```

### On Raspberry Pi 4B

**Via SSH**:
```bash
ssh pi@pi-4b.local
cd turboquant-vla
python scripts/benchmark_octo_pi.py --device cpu
```

**Via rsync**:
```bash
# Sync project to Pi
rsync -av --exclude='.git' ./ pi@pi-4b.local:~/turboquant-vla/

# Run via SSH
ssh pi@pi-4b.local "cd turboquant-vla && python scripts/benchmark_octo_pi.py"
```

## Output Artifacts

- **Latency logs**: `results/latency_*.csv` (timestamped per-action times)
- **Memory logs**: `results/memory_*.csv` (peak usage, timeline)
- **Accuracy logs**: `results/accuracy_*.json` (action MSE, task success)
- **Profiling**: `profiles/` (detailed layer-by-layer breakdown)
- **Reports**: `docs/` (summary benchmarks and analysis)

## Common Flags

```
--device [cpu|cuda|mps]      Hardware to run on
--batch-size N               Inference batch size (default: 1)
--num-runs N                 Number of benchmark runs (default: 10)
--warmup N                   Warmup iterations (default: 2)
--save-profile               Save detailed profiling output
--output-dir DIR             Override output directory
--quiet                      Suppress verbose logging
```

## Dependencies

Scripts assume:
- `turboquant_kv` installed (from package or local)
- Model available on HuggingFace or local disk
- PyTorch + appropriate hardware (CUDA for GPU, CPU-only for Pi)
- Optional: ONNX Runtime for Phase 3+

## Troubleshooting

**Script fails to load model**:
- Check HuggingFace token in `.env` (if private model)
- Verify model exists: `huggingface-cli repo-info <MODEL_ID>`

**Memory error on Pi**:
- Reduce `--batch-size` (default: 1 is already minimal)
- Kill background processes: `killall python`
- Check available RAM: `free -h`

**Slow inference on Pi**:
- Check CPU throttling: `vcgencmd get_throttled` (should be 0x0)
- Add heatsink or fan if temp > 80°C
- Reduce inference frequency if running on-robot

## Adding New Benchmarks

1. Create `scripts/benchmark_YOUR_FEATURE.py`
2. Follow template structure:
   ```python
   import argparse
   from pathlib import Path
   
   def main(args):
       # Load model
       # Run benchmark
       # Collect metrics
       # Save results
   
   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--device", default="cpu")
       # Add more args
       args = parser.parse_args()
       main(args)
   ```
3. Document output format
4. Add to `run_all_benchmarks.sh`
