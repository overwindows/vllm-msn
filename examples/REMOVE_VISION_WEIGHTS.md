# Removing Vision Weights from Gemma 4 for Text-Only Inference

## Summary

**Potential savings: ~1.5-2 GB** by removing unused vision components.

---

## Current Situation

### Model Architecture:
```
Gemma 4 26B "A4B" = All-in-one model with:
├─ Text encoder: 30 layers, 2816 hidden (23.5 GB) ← What you use ✓
├─ Vision encoder: 27 layers, 1152 hidden (~1.5 GB) ← NOT USED ✗
└─ Cross-modal projection layers (~0.5 GB) ← NOT USED ✗

Total model size: ~49 GB on disk (BF16)
Text-only needed: ~47 GB on disk (BF16)
Savings: ~2 GB on disk, ~1-1.5 GB in GPU memory
```

### Weight Distribution:
```
model-00001-of-00002.safetensors (46.48 GB):
├─ 599 text tensors (embeddings, MoE experts, attention)
├─ 356 vision tensors (vision_tower.*, embed_vision.*) ← Can remove
└─ Mixed in same file (requires extraction)

model-00002-of-00002.safetensors (1.59 GB):
└─ 58 text tensors (final layers, output projection)
    └─ No vision weights ✓
```

### Vision Components in model-00001:
```python
Vision tower layers (27 layers):
- model.vision_tower.encoder.layers.*.input_layernorm.weight
- model.vision_tower.encoder.layers.*.mlp.*.weight
- model.vision_tower.encoder.layers.*.self_attn.*.weight
- model.embed_vision.embedding_projection.weight
- etc.

Total: 356 tensors ≈ 1.5-2 GB
```

---

## Does vLLM Load Vision Weights?

### Quick Check:

```bash
# Test if vision weights are loaded
cd /nvmedata/chenw/vllm-ra/examples

# Add verbose logging
export VLLM_LOG_LEVEL=INFO
export VLLM_TRACE_FUNCTION=1

# Run a single inference and check loaded weights
python3 << 'PYTHON_EOF'
from vllm import LLM

# Initialize model (watch memory usage)
llm = LLM(
    model="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it",
    dtype="bfloat16",
    quantization="fp8",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
)

# Check which parameters were loaded
print("\nLoaded parameter groups:")
for name, param in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters():
    if 'vision' in name.lower():
        print(f"  VISION: {name} - {param.numel() * 2 / 1024**2:.1f} MB")
        break
else:
    print("  ✓ No vision parameters found (optimized away)")
PYTHON_EOF
```

### Expected Behavior:

**vLLM 0.10+ should automatically skip vision weights** for text-only models:
- Detects you're not passing images
- Doesn't load vision_tower weights
- Saves ~1.5 GB GPU memory ✓

**But if vLLM loads them anyway:**
- You're wasting ~1.5 GB GPU memory
- Need to manually remove vision weights

---

## Option 1: Check if Already Optimized (Recommended First Step)

```bash
# Monitor memory during model loading
watch -n 1 nvidia-smi &

# Load model and check memory
cd /nvmedata/chenw/vllm-ra/examples
source /root/miniconda3/bin/activate vllm

python3 << 'PYTHON_EOF'
import torch
from vllm import LLM

print("Memory before loading:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

llm = LLM(
    model="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it",
    dtype="bfloat16",
    quantization="fp8",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
)

print("\nMemory after loading:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Check for vision weights
has_vision = False
for name, _ in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters():
    if 'vision' in name.lower():
        has_vision = True
        break

if has_vision:
    print("\n⚠️  Vision weights ARE loaded (wasting ~1.5GB)")
    print("   → Consider removing vision weights")
else:
    print("\n✓ Vision weights NOT loaded (already optimized)")
    print("   → No action needed")
PYTHON_EOF
```

**If vision weights are NOT loaded**: You're already good! ✓
**If vision weights ARE loaded**: Proceed to Option 2 or 3.

---

## Option 2: Create Text-Only Model (Permanent Solution)

If vLLM loads vision weights unnecessarily, create a text-only variant:

### Step 1: Extract Text-Only Weights

```python
#!/usr/bin/env python3
# save as: create_text_only_model.py

from safetensors.torch import load_file, save_file
import json
import shutil
import os

model_path = "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
output_path = "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only"

print("Creating text-only variant...")
os.makedirs(output_path, exist_ok=True)

# Process each safetensors file
for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    input_file = os.path.join(model_path, shard)
    output_file = os.path.join(output_path, shard)

    print(f"\nProcessing {shard}...")

    # Load weights
    weights = load_file(input_file)

    # Filter out vision weights
    text_weights = {}
    vision_weights_removed = 0

    for key, tensor in weights.items():
        if 'vision' in key.lower() or 'image' in key.lower():
            vision_weights_removed += 1
            print(f"  Removing: {key}")
        else:
            text_weights[key] = tensor

    print(f"  Kept: {len(text_weights)} tensors")
    print(f"  Removed: {vision_weights_removed} vision tensors")

    # Save text-only weights
    save_file(text_weights, output_file)

    # Calculate size savings
    original_size = os.path.getsize(input_file) / (1024**3)
    new_size = os.path.getsize(output_file) / (1024**3)
    savings = original_size - new_size

    print(f"  Original: {original_size:.2f} GB")
    print(f"  New: {new_size:.2f} GB")
    print(f"  Savings: {savings:.2f} GB")

# Copy config and modify
print("\nUpdating config...")
with open(os.path.join(model_path, "config.json"), "r") as f:
    config = json.load(f)

# Remove vision config
if "vision_config" in config:
    config["vision_config"] = None
if "audio_config" in config:
    config["audio_config"] = None

# Save modified config
with open(os.path.join(output_path, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Copy other files
for filename in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "model.safetensors.index.json", "chat_template.jinja",
                 "processor_config.json", ".gitattributes"]:
    src = os.path.join(model_path, filename)
    dst = os.path.join(output_path, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {filename}")

print("\n✓ Text-only model created at:")
print(f"  {output_path}")
print("\nTo use:")
print(f"  --model_path {output_path}")
```

### Step 2: Run the Script

```bash
cd /nvmedata/chenw/vllm-ra/examples
source /root/miniconda3/bin/activate vllm

# Install safetensors if needed
pip install safetensors

# Create text-only model
python3 create_text_only_model.py
```

### Step 3: Update Your Configuration

```bash
# In vllm_gemma4_moe_fp8_mtp.sh
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only  # Changed!
ASSISTANT_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant

# Everything else stays the same
```

### Expected Savings:
```
Disk space: ~2 GB saved
GPU memory: ~1-1.5 GB saved
Loading time: ~5-10 seconds faster
```

---

## Option 3: Use vLLM Model Loading Options (If Available)

Check if vLLM has text-only loading flags:

```python
# In llm_analyzer_gemma4_moe_fp8_mtp.py

engine_args = AsyncEngineArgs(
    model=model_path,
    # ... other args ...

    # Try these (may not be available in all versions):
    skip_tokenizer_init=False,
    load_format="safetensors",  # Explicit format
    # enforce_eager=False,  # Already set

    # Potential future flags (check vLLM docs):
    # multimodal=False,  # Disable multimodal support
    # text_only=True,    # Load text components only
)
```

Check vLLM version for text-only options:
```bash
python3 -c "from vllm import __version__; print(__version__)"
# Then check: https://docs.vllm.ai/en/latest/
```

---

## Option 4: Use Symlinks (Quick Hack)

If you just want to test without vision weights:

```bash
# Create a test directory with symlinks
mkdir -p /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-test
cd /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-test

# Symlink everything except model files
for f in ../gemma-4-26B-A4B-it/*; do
    if [[ ! $f =~ \.safetensors$ ]]; then
        ln -s "$f" .
    fi
done

# Copy only model-00002 (text-only shard)
cp ../gemma-4-26B-A4B-it/model-00002-of-00002.safetensors .

# Manually create a minimal model-00001 (would need scripting)
# This is complex - use Option 2 instead
```

**Verdict**: Too complex, use Option 2 instead.

---

## Recommended Approach

### Phase 1: Verify (5 minutes)
```bash
# Check if vLLM already optimizes away vision weights
./check_vision_loaded.sh  # Script from Option 1
```

### Phase 2A: If vision NOT loaded
```
✓ You're already optimized!
✓ No action needed
✓ Vision weights are skipped automatically
```

### Phase 2B: If vision IS loaded
```bash
# Create text-only model (10 minutes)
python3 create_text_only_model.py

# Update config to use text-only model
sed -i 's|gemma-4-26B-A4B-it|gemma-4-26B-A4B-it-text-only|g' \
    vllm_gemma4_moe_fp8_mtp.sh

# Test with text-only model
./vllm_gemma4_moe_fp8_mtp.sh
```

---

## Memory Impact Analysis

### Before (with vision):
```
Model weights:        22 GB (text + vision)
KV cache:              8 GB
CUDA graphs:           5 GB
Activations:           3 GB
──────────────────────────
Total:                38 GB / 40 GB (95%)
Headroom:              2 GB (tight!)
```

### After (text-only):
```
Model weights:        20.5 GB (text only)  ← -1.5 GB
KV cache:              8 GB
CUDA graphs:           5 GB
Activations:           3 GB
──────────────────────────
Total:                36.5 GB / 40 GB (91%)
Headroom:              3.5 GB (better!)
```

### Benefits:
```
✓ 1.5 GB more headroom
✓ Could enable larger batches (128 → 160)
✓ Faster model loading (-5-10 seconds)
✓ Less disk space (-2 GB)
✓ Cleaner deployment (no unused code)
```

---

## Potential Issues

### Issue 1: Model Loading Errors

**Error**: `KeyError: 'vision_tower.encoder.layers.0...'`

**Solution**: Make sure model.safetensors.index.json is updated:
```python
# After removing vision weights, update index
import json

with open("model.safetensors.index.json", "r") as f:
    index = json.load(f)

# Remove vision weight references
weight_map = index["weight_map"]
index["weight_map"] = {k: v for k, v in weight_map.items()
                       if 'vision' not in k.lower()}

with open("model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)
```

### Issue 2: vLLM Expects Vision Components

**Error**: `AttributeError: 'Gemma4ForConditionalGeneration' has no attribute 'vision_tower'`

**Solution**: You might need to modify the model architecture:
```python
# This is advanced - might require vLLM source modification
# Easier to ensure vLLM version supports text-only models
```

### Issue 3: Assistant Model Still Has Vision

**Note**: The assistant model is separate and smaller.
Check if it also has vision components:
```bash
ls -lh /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant/*.safetensors
# If single file < 1GB, likely text-only already ✓
```

---

## Testing

After removing vision weights:

```bash
# Test 1: Can it load?
python3 -c "
from vllm import LLM
llm = LLM(model='/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only')
print('✓ Model loaded successfully')
"

# Test 2: Can it generate?
python3 -c "
from vllm import LLM
llm = LLM(model='/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only')
output = llm.generate('Hello, how are you?', max_tokens=20)
print('✓ Generation works:', output[0].outputs[0].text)
"

# Test 3: Memory usage
python3 << 'EOF'
import torch
from vllm import LLM

print(f"Before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
llm = LLM(model='/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only',
          gpu_memory_utilization=0.75)
print(f"After: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
# Should be ~1.5 GB less than with vision
EOF
```

---

## Summary

**Recommended action:**

1. **First**: Check if vLLM already skips vision weights (Option 1)
   - If yes: No action needed ✓
   - If no: Proceed to step 2

2. **If needed**: Create text-only model variant (Option 2)
   - Saves 1.5 GB GPU memory
   - Takes 10 minutes
   - One-time setup

3. **Update** your configuration to use text-only model

4. **Test** to ensure everything works

**Expected benefit:**
- Memory: +1.5 GB headroom (3.75% more available)
- Could enable batch_size: 128 → 160-180
- Throughput: +20-30% from larger batches
- Total gain: Worth doing if memory is tight!

Your A100 40GB is constrained, so every GB counts. This optimization could allow larger batches and better throughput! 🎯
