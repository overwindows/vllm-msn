# Attention Patterns: Dense vs Sparse

## Quick Answer

**Yes, sliding window attention IS sparse attention.**

Specifically, it's a **local sparse attention pattern** where each token only attends to a fixed window of nearby tokens.

---

## Visual Comparison

### 1. Dense (Full) Attention

```
Attention matrix for 8 tokens (full O(N²)):

Query Token →
     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │  Token 0 attends to all 8 tokens
1 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │  Token 1 attends to all 8 tokens
2 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │  Token 2 attends to all 8 tokens
3 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │  ...
4 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
5 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
6 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
7 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
  └──────────────────────────┘
  ↑
  Key Token

Total computations: 8 × 8 = 64
Sparsity: 0% (dense)
Complexity: O(N²)
```

### 2. Sliding Window Attention (window=3)

```
Attention matrix with window=3:

Query Token →
     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ·  ·  ·  ·  ·  ·  ·  │  Token 0: only self
1 │ ✓  ✓  ·  ·  ·  ·  ·  ·  │  Token 1: [0,1]
2 │ ✓  ✓  ✓  ·  ·  ·  ·  ·  │  Token 2: [0,1,2]
3 │ ·  ✓  ✓  ✓  ·  ·  ·  ·  │  Token 3: [1,2,3] ← window slides
4 │ ·  ·  ✓  ✓  ✓  ·  ·  ·  │  Token 4: [2,3,4]
5 │ ·  ·  ·  ✓  ✓  ✓  ·  ·  │  Token 5: [3,4,5]
6 │ ·  ·  ·  ·  ✓  ✓  ✓  ·  │  Token 6: [4,5,6]
7 │ ·  ·  ·  ·  ·  ✓  ✓  ✓  │  Token 7: [5,6,7]
  └──────────────────────────┘

Pattern: Diagonal band (banded matrix)
Total computations: 1+2+3+3+3+3+3+3 = 21
Sparsity: 67% sparse (21/64 = 33% dense)
Complexity: O(N × window)
```

### 3. Gemma 4's Pattern (window=1024)

```
For long sequence (e.g., 8192 tokens):

Dense attention:
  8192 × 8192 = 67,108,864 computations

Sliding window (1024):
  First 1024 tokens: 1 + 2 + 3 + ... + 1024 = 524,800
  Remaining 7168 tokens: 7168 × 1024 = 7,340,032
  Total: ≈ 7,864,832 computations

Sparsity: 88.3% sparse!
  Only 11.7% of attention matrix is computed
```

---

## Types of Sparse Attention

Sliding window is **one type** of sparse attention. Here are others:

### 1. Local/Sliding Window (Gemma 4 uses this!)

```
Pattern: Diagonal band

     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ·  ·  ·  ·  ·  ·  ·  │
1 │ ✓  ✓  ·  ·  ·  ·  ·  ·  │
2 │ ✓  ✓  ✓  ·  ·  ·  ·  ·  │
3 │ ·  ✓  ✓  ✓  ·  ·  ·  ·  │
4 │ ·  ·  ✓  ✓  ✓  ·  ·  ·  │
5 │ ·  ·  ·  ✓  ✓  ✓  ·  ·  │
6 │ ·  ·  ·  ·  ✓  ✓  ✓  ·  │
7 │ ·  ·  ·  ·  ·  ✓  ✓  ✓  │
  └──────────────────────────┘

Properties:
✓ Captures local context
✓ O(N × window) complexity
✓ Good for sequential data (text, time series)
✗ Misses long-range dependencies

Used by: Gemma 4, Longformer (partially), Mistral
```

### 2. Strided Attention

```
Pattern: Every k-th token (e.g., k=2)

     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ·  ✓  ·  ✓  ·  ✓  ·  │  Attend to 0,2,4,6
1 │ ·  ✓  ·  ✓  ·  ✓  ·  ✓  │  Attend to 1,3,5,7
2 │ ✓  ·  ✓  ·  ✓  ·  ✓  ·  │
3 │ ·  ✓  ·  ✓  ·  ✓  ·  ✓  │
4 │ ✓  ·  ✓  ·  ✓  ·  ✓  ·  │
5 │ ·  ✓  ·  ✓  ·  ✓  ·  ✓  │
6 │ ✓  ·  ✓  ·  ✓  ·  ✓  ·  │
7 │ ·  ✓  ·  ✓  ·  ✓  ·  ✓  │
  └──────────────────────────┘

Properties:
✓ Captures some long-range dependencies
✓ O(N²/k) complexity
✗ Fixed stride may miss important patterns

Used by: Sparse Transformers (OpenAI, 2019)
```

### 3. Block-Sparse Attention

```
Pattern: Block structure (blocks of size 2)

     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ✓  ·  ·  ·  ·  ·  ·  │  Block 0 attends to Block 0
1 │ ✓  ✓  ·  ·  ·  ·  ·  ·  │
2 │ ·  ·  ✓  ✓  ·  ·  ·  ·  │  Block 1 attends to Block 1
3 │ ·  ·  ✓  ✓  ·  ·  ·  ·  │
4 │ ·  ·  ·  ·  ✓  ✓  ·  ·  │  Block 2 attends to Block 2
5 │ ·  ·  ·  ·  ✓  ✓  ·  ·  │
6 │ ·  ·  ·  ·  ·  ·  ✓  ✓  │  Block 3 attends to Block 3
7 │ ·  ·  ·  ·  ·  ·  ✓  ✓  │
  └──────────────────────────┘

Properties:
✓ Hardware-friendly (aligned blocks)
✓ Good for structured data
✗ May miss cross-block patterns

Used by: BlockBERT, FlashInfer (for efficiency)
```

### 4. Random Sparse Attention

```
Pattern: Random subset of attention

     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ·  ✓  ·  ·  ✓  ·  ·  │  Random 3 positions
1 │ ·  ✓  ·  ✓  ·  ·  ✓  ·  │  Random 3 positions
2 │ ✓  ·  ✓  ·  ✓  ·  ·  ·  │
3 │ ·  ·  ·  ✓  ·  ✓  ·  ✓  │
4 │ ·  ✓  ·  ·  ✓  ·  ✓  ·  │
5 │ ✓  ·  ·  ✓  ·  ✓  ·  ·  │
6 │ ·  ·  ✓  ·  ·  ·  ✓  ✓  │
7 │ ·  ✓  ·  ·  ✓  ·  ·  ✓  │
  └──────────────────────────┘

Properties:
✓ Can capture diverse patterns
✓ Theoretical coverage guarantees
✗ Non-deterministic
✗ Hard to optimize in hardware

Used by: BigBird (random attention component)
```

### 5. Hybrid Pattern (Gemma 4's Full Picture!)

```
Gemma 4 combines TWO patterns:

Layer 0-4: Sliding Window (sparse)
     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ·  ·  ·  ·  ·  ·  ·  │
1 │ ✓  ✓  ·  ·  ·  ·  ·  ·  │
2 │ ✓  ✓  ✓  ·  ·  ·  ·  ·  │
3 │ ·  ✓  ✓  ✓  ·  ·  ·  ·  │  Sparse!
4 │ ·  ·  ✓  ✓  ✓  ·  ·  ·  │
5 │ ·  ·  ·  ✓  ✓  ✓  ·  ·  │
6 │ ·  ·  ·  ·  ✓  ✓  ✓  ·  │
7 │ ·  ·  ·  ·  ·  ✓  ✓  ✓  │
  └──────────────────────────┘

Layer 5: Full Attention (dense)
     0  1  2  3  4  5  6  7
  ┌──────────────────────────┐
0 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
1 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
2 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
3 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │  Dense!
4 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
5 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
6 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
7 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  │
  └──────────────────────────┘

Best of both worlds:
✓ 83% layers are sparse (efficient)
✓ 17% layers are dense (global context)
```

---

## Terminology Clarification

### "Sparse Attention" can mean:

1. **Pattern-level sparsity** (what we usually mean)
   - Not all tokens attend to all other tokens
   - Examples: sliding window, strided, block-sparse
   - **Gemma 4's sliding window IS this type** ✓

2. **Weight-level sparsity** (less common, different meaning)
   - Attention weights themselves are sparse (pruned)
   - Some attention connections permanently removed
   - Not what Gemma 4 does

3. **Activation-level sparsity** (MoE-related)
   - Sparse expert activation (like Gemma 4's MoE)
   - Different from attention sparsity
   - Gemma 4 has BOTH sparse attention AND sparse MoE!

### What Gemma 4 Has:

```
Gemma 4 26B MoE has TWO types of sparsity:

1. Sparse Attention (sliding window)
   └─ 83% of layers use local/sparse attention pattern
   └─ Only attend to last 1024 tokens (not full sequence)

2. Sparse MoE (top-k routing)
   └─ 128 experts per layer
   └─ Only 8 active per token (93.75% sparse!)

Combined sparsity:
- Attention: 88% sparse (for long sequences)
- MoE: 93.75% sparse (expert activation)
- Total effective sparsity: ~99% of possible computations skipped!
```

---

## Why Sparse Attention Matters

### Memory Complexity:

```
Sequence length N = 8192 tokens

Dense attention:
  Attention matrix: N × N = 8192² = 67,108,864 elements
  Memory: 67M × 2 bytes (FP16) = 134 MB per layer
  30 layers: 30 × 134 MB = 4 GB just for attention!

Sparse sliding window (1024):
  Attention matrix: N × window = 8192 × 1024 = 8,388,608 elements
  Memory: 8.4M × 2 bytes = 17 MB per layer
  30 layers: 30 × 17 MB = 510 MB

Savings: 4 GB → 0.5 GB = 87.5% less memory! ✓
```

### Compute Complexity:

```
For autoregressive generation (generating 1 token at a time):

Dense attention:
  New token attends to all N previous tokens
  Compute: O(N) per token generated
  Total for M tokens: O(N × M)

Sparse sliding (window=W):
  New token attends to last W tokens only
  Compute: O(W) per token (W is constant!)
  Total for M tokens: O(W × M) = O(M)

Example: N=8192, W=1024, M=100 new tokens
  Dense: 8192 × 100 = 819,200 ops
  Sparse: 1024 × 100 = 102,400 ops
  Speedup: 8x faster! ✓
```

---

## FlashInfer and Sparse Attention

**Why FlashInfer is good for Gemma 4's sliding window:**

```
FlashInfer specialization:
1. Block-sparse kernels
   └─ Sliding window is naturally block-structured
   └─ Window of 1024 = 64 blocks of 16 tokens

2. PagedAttention integration
   └─ vLLM allocates KV cache in pages/blocks
   └─ Sliding window maps perfectly to paged blocks

3. Efficient memory access
   └─ Only loads relevant blocks (last 1024 tokens)
   └─ Skips blocks outside window (no memory waste)

4. Hardware optimization
   └─ Coalesced memory reads for block access
   └─ Better cache locality for local attention
```

**Visual: How FlashInfer handles sliding window:**

```
KV cache in vLLM (paged):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│Block │Block │Block │Block │Block │Block │
│  0   │  1   │  2   │  3   │  4   │  5   │
│ [0-  │ [16- │ [32- │ [48- │ [64- │ [80- │
│  15] │  31] │  47] │  63] │  79] │  95] │
└──────┴──────┴──────┴──────┴──────┴──────┘

For token 95 (in Block 5) with window=48:
  Need: tokens [48-95]
  Blocks needed: 3, 4, 5  (3 blocks)
  Blocks skipped: 0, 1, 2  (3 blocks) ← FlashInfer efficiently skips!

FlashInfer:
✓ Loads only blocks 3,4,5
✓ Computes attention only on these blocks
✓ Saves memory bandwidth
✓ Faster than loading all 6 blocks
```

---

## Summary Table

| Property | Dense Attention | Sliding Window (Sparse) |
|----------|----------------|-------------------------|
| **Pattern** | Full N×N | Diagonal band (width W) |
| **Sparsity** | 0% (all computed) | ~(N-W)/N % for large N |
| **Memory** | O(N²) | O(N × W) |
| **Compute** | O(N²) | O(N × W) |
| **Context** | Global (all tokens) | Local (last W tokens) |
| **Long-range** | Yes | No (limited to W) |
| **Efficiency** | Low for large N | High for large N |
| **Example** | GPT-2, LLaMA (all layers), Gemma 4 (5 layers) | Gemma 4 (25 layers), Longformer, Mistral |

**Gemma 4's Innovation:**
Uses BOTH sparse (83%) and dense (17%) to get:
✓ Efficiency of sparse (memory/compute)
✓ Global context of dense (long-range dependencies)

---

## Conclusion

**Yes, sliding window attention is sparse attention!**

Specifically:
- **Pattern**: Local/banded sparse attention
- **Sparsity**: 88% for long sequences (Gemma 4's window=1024)
- **Complexity**: O(N × window) instead of O(N²)
- **Benefit**: 8x less memory and compute for typical workloads

**Gemma 4's sliding window makes it:**
- Memory-efficient (can handle 262K context)
- Fast for inference (O(N) instead of O(N²))
- Compatible with vLLM's PagedAttention
- Ideal for FlashInfer optimization (block-sparse)

The hybrid design (83% sparse + 17% dense) is what makes Gemma 4 both efficient AND capable of modeling long-range dependencies!
