# Zcore Implementation Plan

> **Vision**: A fast, well-documented tensor library in pure Zig -> NumPy/PyTorch for Zig. Every line is written for
> beginners to understand. First in a line of planned data-science libraries (dataframes, plotting) for Zig.

## Design Philosophy

1. **Zero deps**: No external dependencies beyond the Zig standard library.
2. **Educational by default**: Every function is documented; every design decision is explained in comments.
3. **Progressive optimization**: Start correct, then make fast. Each operation has a naive reference impl that can be
   swapped for an optimized SIMD/multithreaded version later.
4. **IEEE 754 compliance**: Follow IEEE 754 for floating-point (NaN propagation, +-0, +-inf, rounding).
5. **No AI-generated code**: Per the Zig community's norms, not even documentation should be AI generated.

## Versioning Scheme

| Prefix | Meaning |
|--------|---------|
| `v0.0.x` | Core tensor skeleton -> shape, strides, memory, dtypes, basic constructors, indexing |
| `v0.1.x` | Shape/view manipulation -> reshape, transpose, slice, flatten, permute, expand, squeeze, iterator |
| `v0.2.x` | Arithmetic & broadcasting -> element-wise ops, comparisons, reductions |
| `v0.3.x` | Linear algebra -> matmul (progressive), decompositions, norms, determinants |
| `v0.4.x` | Random, serialization, utils -> distributions, save/load, seeding |
| `v0.5.x` | Neural network primitives -> convolution, pooling, loss functions, activations |
| `v0.6.x` | Autograd -> computational graph, gradient tape, backward pass |
| `v0.7.x` | Performance -> SIMD vectorization, multithreading, BLAS integration |
| `v1.0`   | Stable API -> benchmarks, docs, packaging, README, examples |

---
## Phase 1: Core Tensor -> v0.0.x

**Goal**: A working `Tensor(T)` that owns memory, supports basic constructors, and can be indexed.

### Implementation

- [x] Tensor struct with `_data`, `_shape`, `_strides`, `_allocator`, `_owns_memory`
- [x] Stride computation (row-major, C-style)
- [x] `init(allocator, shape)` -> allocates data + shape + strides
- [x] `destroy()` -> frees owned memory
- [x] Supported types: `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`, `bool`
- [x] Comptime gate to reject non-numeric types (with `@typeInfo`)
- [ ] NaN handling policy -> document which NaN we use, how comparisons behave
- [x] `fill(value)` -> set every element
- [x] `zeroes(allocator, shape)` -> convenience for zero-initialized
- [x] `ones(allocator, shape)` -> convenience for one-initialized
- [ ] `empty(allocator, shape)` -> allocate without initializing (unsafe; for perf)
- [ ] `full(allocator, shape, value)` -> allocate & fill
- [ ] `fromSlice(allocator, shape, data)` -> adopt existing data
- [x] `get(indices)` -> bounds-checked element access
- [x] `getUnchecked(indices)` -> no bounds checking
- [x] `set(indices, value)` -> bounds-checked write
- [x] `setRow`, `setCol`, `setWhole`
- [x] `debug_print()` -> recursive, multi-dim formatting

### Testing (v0.0.x)

- [x] Stride correctness for 1d, 2d, 3d
- [x] Element count
- [x] `zeroes` fills with 0
- [x] `set` / `get` / `at` round-trip
- [ ] `fromSlice` correctness
- [x] Type rejection at comptime for non-numeric types
- [ ] Edge cases: 0-d tensor (scalar), 0-size dimension

### Infrastructure

- [x] Build system (`build.zig`, `build.zig.zon`)
- [x] Module exposed as `zcore`
- [x] `zig build test` runs unit tests
- [x] `zig build example1` runs a demo

---

## Phase 2: Shape Manipulation -> v0.1.x

**Goal**: All common view operations -> no data copies, just stride/shape tricks.

### Learning

- [ ] How views work in NumPy / PyTorch (no-copy `__array_interface__`)
- [ ] Stride manipulation for transpose / permute / slice

### Implementation

| Operation | Description | Notes |
|-----------|-------------|-------|
| `reshape(new_shape)` | Change shape if element count matches | May need copy if non-contiguous |
| `transpose(dims?)` | General n-d transpose; default = reverse | Already done for 2d |
| `slice(ranges)` | Extract view over hyper-rectangle | Already done for 1st dim |
| `flatten()` | Collapse to 1-d contiguous view | |
| `permute(dims)` | Arbitrary axis reordering (stride swap) | |
| `unsqueeze(dim)` / `expandDim(dim)` | Add a dimension of size 1 | |
| `squeeze(dim?)` | Remove dims of size 1 | |
| `iterator()` | Walk elements in storage order or logical order | Needed for efficient loops |

### Contiguity

- [ ] Add `isContiguous()` -> checks if strides match canonical row-major
- [ ] Add `contiguous()` -> returns a contiguous copy if needed; no-op otherwise
- [ ] `reshape` on a non-contiguous tensor must copy first

### Testing (v0.1.x)

- [ ] Every op with correct shape/stride after transform
- [ ] Non-contiguous `reshape` copies
- [ ] `flatten` on transposed tensor
- [ ] Slice bounds and zero-size slices
- [ ] `squeeze` / `unsqueeze` round-trip

---

## Phase 3: Arithmetic & Broadcasting -> v0.2.x

**Goal**: Element-wise operations, broadcasting, comparisons, and reductions.

### Learning

- [ ] Broadcasting rules (NumPy: align trailing dims, treat 1 as broadcast)

### Broadcasting

- [ ] `broadcastShapes(a, b)` -> static shape inference
- [ ] `broadcastTo(target_shape)` -> returns a view with expanded strides (stride=0 for broadcast dims)
- [ ] Fallback: if view is not possible, materialize a copy

### Element-wise Arithmetic

| Operation | Description | Notes |
|-----------|-------------|-------|
| `add`, `sub`, `mul`, `div` (scalar & tensor) | `@` operator overloads | |
| `neg` | Unary negation | |
| `pow` (scalar & tensor) | Exponentiation | Integer pow for int types |
| `sqrt`, `rsqrt` | Square root / reciprocal sqrt | |
| `abs` | Absolute value | |
| `exp`, `log`, `log2`, `log10` | Exponents & logarithms | |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan` | Trigonometry | |
| `sinh`, `cosh`, `tanh` | Hyperbolic | |
| `floor`, `ceil`, `round`, `trunc` | Rounding | |
| `sigmoid`, `relu`, `gelu` | Common activations | Quick if no autograd yet |

### Comparison & Logical

| Operation | Description |
|-----------|-------------|
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | Element-wise comparison → bool tensor |
| `all`, `any` | Reduce over axis |
| `where(condition, a, b)` | Element-wise select |

### Reductions

| Operation | Description |
|-----------|-------------|
| `sum(axis?)` | Sum over optional axis |
| `mean(axis?)` | Arithmetic mean |
| `min(axis?)`, `max(axis?)` | Extremal values |
| `argmin(axis?)`, `argmax(axis?)` | Indices of extremal values |
| `var(axis?)`, `std(axis?)` | Variance / standard deviation |
| `prod(axis?)` | Product of elements |
| `cumsum`, `cumprod` | Cumulative reductions |

### NaN / Infinity Rules (IEEE 754)

- [ ] Document: NaN is quiet NaN (qNaN) from `@as(T, @bitCast(@maxValue(T)))`
- [ ] Arithmetic: NaN propagates per IEEE 754
- [ ] Comparisons: `NaN != NaN` (IEEE)
- [ ] Reductions: `sum` with NaN → NaN; `min`/`max` skip NaN? Document choice.

### Testing (v0.2.x)

- [ ] Property tests: `a + b == b + a` (commutativity)
- [ ] Broadcasting with all edge cases (different ranks, dim=1, dim mismatch → error)
- [ ] NaN propagation in every arithmetic op
- [ ] Reduction with axis = null (global) vs axis = 0, 1, ...
- [ ] Comparison → bool tensor → used in `where`

---

## Phase 4: Linear Algebra -> v0.3.x

**Goal**: Matrix multiplication and common decompositions.

### Learning

- [ ] Naive O(n³) matmul → cache-blocked → SIMD → multithreaded
- [ ] Strassen O(n^2.81) (optional, big matrices only)
- [ ] BLIS / BLAS conventions

### Matrix Multiplication

| Step | Description |
|------|-------------|
| `matmul(a, b)` -> naive | Triple loop, reference impl |
| `matmul(a, b)` -> tiled | Cache-blocked (B += 3× perf) |
| `matmul(a, b)` -> SIMD | Use `@Vector` for inner loop |
| `matmul(a, b)` -> multithreaded | Parallel outer loops |
| Batch matmul | `bmm` for 3-d tensors |

### Decompositions & LA Ops

| Operation | Description |
|-----------|-------------|
| `dot(a, b)` | Inner product (1-d) |
| `cross(a, b)` | Cross product (3-d vectors) |
| `norm(a, order?)` | Vector / matrix norms |
| `det(a)` | Determinant (via LU) |
| `inv(a)` | Matrix inverse (via LU) |
| `lu(a)` | LU decomposition with partial pivoting |
| `qr(a)` | QR decomposition (Gram–Schmidt / Householder) |
| `svd(a)` | Singular value decomposition |
| `cholesky(a)` | Cholesky decomposition (positive-definite only) |
| `eig(a)` | Eigenvalue decomposition (symmetric for now) |

### Testing (v0.3.x)

- [ ] `matmul` identity: `A @ I == A` and `I @ A == A`
- [ ] `matmul` associative: `(A @ B) @ C == A @ (B @ C)` (within float tolerance)
- [ ] `inv(A) @ A == I`
- [ ] `det` correctness on 2×2, 3×3
- [ ] `svd` reconstruction: `U @ S @ V^T ≈ A`
- [ ] Benchmark comparisons across matmul impls (naive vs tiled vs SIMD)

---

## Phase 5: Random & Serialization -> v0.4.x

**Goal**: Random number generation, shuffling, and tensor I/O.

### Learning

- [ ] PRNGs: Xoshiro256**, PCG, ChaCha -> pros and cons
- [ ] NumPy's random API design

### Random

| Feature | Description |
|---------|-------------|
| `seed(val)` | Seed the global RNG state |
| `rand(shape)` | Uniform (0, 1) |
| `randn(shape)` | Normal via Box–Muller / Ziggurat |
| `randint(low, high, shape)` | Integer uniform |
| `bernoulli(p, shape)` | Bernoulli trials |
| `uniform(low, high, shape)` | Custom range |
| `shuffle(tensor)` | Shuffle along first axis |
| `permutation(n)` | Random permutation of 0..n |

### Serialization

| Feature | Description |
|---------|-------------|
| `save(tensor, path)` | Write shape + dtype + data to binary file or JSON |
| `load(allocator, path)` | Read back a tensor |
| Format: custom binary (fast), NPY (interop), or both |

### Seeding & Reproducibility

- [ ] Deterministic by default when seeded
- [ ] Thread-safe RNG state

### Testing (v0.4.x)

- [ ] Statistical: chi-squared test for uniform distributions
- [ ] Serialization round-trip: save → load → identical tensor
- [ ] Seed reproducibility: same seed → same sequence

---

## Phase 6: Neural Network Primitives -> v0.5.x

**Goal**: Ops needed to build and train simple neural networks.

### Learning

- [ ] Convolution as im2col + matmul
- [ ] Backpropagation maths for each op

### Convolution

| Feature | Description |
|---------|-------------|
| `conv2d(input, kernel, stride, padding)` | Naive reference impl |
| `conv2d` -> im2col | Via `im2col` + `matmul` (much faster) |
| `conv_transpose2d` | Transposed convolution |

### Pooling

| Feature | Description |
|---------|-------------|
| `max_pool2d(input, kernel, stride)` | Max pooling |
| `avg_pool2d(input, kernel, stride)` | Average pooling |
| `global_avg_pool` | Reduce each channel to scalar |

### Loss Functions

| Feature | Description |
|---------|-------------|
| `mse_loss(pred, target)` | Mean squared error |
| `cross_entropy(pred, target)` | Categorical cross-entropy |
| `binary_cross_entropy(pred, target)` | Binary cross-entropy |
| `l1_loss(pred, target)` | Mean absolute error |

### Activation Functions

Activations from Phase 3 (relu, sigmoid, tanh, gelu) are reused here. Add:

| Feature | Description |
|---------|-------------|
| `softmax(dim)` | Softmax over a dimension |
| `log_softmax(dim)` | Log-softmax for numerical stability |
| `leaky_relu`, `elu`, `selu` | Variants |

### Padding

- [ ] `pad(tensor, pad_width, mode)` -> zero, reflect, replicate, circular

### Testing (v0.5.x)

- [ ] `conv2d` manual: known input + kernel → known output
- [ ] `conv2d` gradient: verify against numerical gradient
- [ ] `max_pool2d` indices for unpooling
- [ ] Cross-entropy on logits with `softmax`

---

## Phase 7: Autograd -> v0.6.x

**Goal**: Automatic differentiation -> build a computation graph, forward pass records ops, backward pass computes
gradients.

### Learning

- [ ] PyTorch autograd: how the tape works, gradient accumulation
- [ ] Reverse-mode automatic differentiation
- [ ] Gradient of all ops from v0.2.x–v0.5.x

### Design

| Component | Description |
|-----------|-------------|
| `GradFn` | Gradient function (closure over saved tensors) |
| `Node` | Wraps a tensor + grad + grad_fn |
| `tape` / `graph` | DAG of operations |
| `backward()` | Traverse DAG in topological order, apply chain rule |
| `no_grad()` context | Disable gradient tracking for inference |

### Gradients by Op

For every op in v0.2.x, v0.3.x, v0.5.x, define:
- Forward: produce output + save tensors needed for backward
- Backward: compute gradients w.r.t. inputs using saved tensors

### Testing (v0.6.x)

- [ ] `x^2` → gradient `2x` (manual verification)
- [ ] Chain rule: `sin(exp(x))` gradient
- [ ] Matmul backward: verify against numerical gradient (finite differences)
- [ ] Conv backward: finite-difference verification
- [ ] Gradient accumulation: `.grad` sums on repeated backward

---

## Phase 8: Performance -> v0.7.x

**Goal**: Make everything fast. No API changes -> swap implementations under the hood.

### SIMD Vectorization

| Op | Strategy |
|----|----------|
| Element-wise arithmetic | `@Vector` -> process 4–16 elements at once |
| Reductions (sum, max) | Horizontal reduction via `@reduce` |
| Matmul inner loop | Vectorized dot product |
| `exp`, `log`, `sin` | Polynomial approximation + vectorization |

### Multithreading

| Op | Strategy |
|----|----------|
| Element-wise large tensors | Split dim 0 across threads |
| Matmul (outer loops) | Parallelize over output tile rows |
| Conv | Parallelize over output channels |
| Reductions | Parallel partial results, then combine |

### BLAS Integration

| Feature | Description |
|---------|-------------|
| Optional BLAS backend | If user links BLIS/OpenBLAS/MKL, dispatch matmul/trsm to it |
| Pure Zig fallback | Our best SIMD+tiled impl when no BLAS is available |

### Memory Optimisation

- [ ] Pool allocator for small tensor temporaries
- [ ] Lazy evaluation / op fusion (plan for future -> flag as post-v1.0)
- [ ] In-place ops where safe (`addAssign`, etc.)

### Benchmarking

- [ ] Continuous benchmark suite (`zig build bench`)
- [ ] Track: matmul GFLOPS, conv throughput, element-wise BW
- [ ] Compare against NumPy (baseline) for correctness

### Testing (v0.7.x)

- [ ] SIMD impl must match reference impl bit-exact for f32 (±0 on identical inputs)
- [ ] Threaded impl must be deterministic (or document non-determinism)
- [ ] No data races -> test with ThreadSanitizer

---

## Phase 9: v1.0 -> API Stabilisation

**Goal**: A stable, documented, benchmarked release.

### Documentation

- [ ] Full API reference (auto-generated from doc comments)
- [ ] Migration guide from NumPy function names to zcore
- [ ] Architecture overview (how memory, views, broadcasting work)
- [ ] Examples: linear regression, MNIST CNN, simple transformer

### Packaging

- [ ] `zig build` publishes to Zig package registry
- [ ] Versioned releases with semver
- [ ] Pre-built binaries for common targets

### Quality

- [ ] 90%+ line coverage
- [ ] Fuzz testing for edge cases (index out of bounds, NaN, inf, denormals)
- [ ] `zig fmt` enforced in CI
- [ ] `zig build` with no warnings
- [ ] `zig build test` with all safety checks enabled

### Ecosystem

- [ ] Write a small `zdf` (dataframe) PoC using zcore
- [ ] Write a small `zplot` PoC using zcore
- [ ] Gather community feedback

---

## Milestone Summary

| Version | Theme | Key Deliverable | Depends On |
|---------|-------|----------------|------------|
| v0.0.x | Core Tensor | `Tensor(T)` with init/destroy/index/basic constructors | Zig + allocators mastered |
| v0.1.x | Shape Manipulation | Views: reshape, transpose, permute, slice, iterator | v0.0.x |
| v0.2.x | Arithmetic | Element-wise ops, broadcasting, reductions | v0.1.x |
| v0.3.x | Linear Algebra | matmul, LU, QR, SVD, eig | v0.2.x |
| v0.4.x | Random & I/O | Distributions, save/load | v0.2.x (uses arithmetic) |
| v0.5.x | NN Primitives | conv, pool, loss, activations | v0.2.x, v0.3.x |
| v0.6.x | Autograd | Gradient tape, backward pass | v0.5.x (needs NN ops) |
| v0.7.x | Performance | SIMD, threading, BLAS | v0.6.x (freeze API first) |
| v1.0   | Stable | Docs, benchmarks, package, CI polish | v0.7.x |

---

## Design Decisions (Expanded)

### NaN Handling

- Use **quiet NaN** (`qNaN`) exclusively -> generated via `@bitCast(@maxValue(T))` for float types.
- Arithmetic: IEEE 754 propagation. Any NaN input → NaN output.
- Comparisons: `==` with NaN returns `false`; `!=` returns `true`.
- Reductions: document whether `sum` / `mean` propagate NaN or skip. Preference: propagate to match IEEE.

### Memory Layout

- **Row-major (C-style)**: strides computed as `strides[i] = product of shape[i+1..]`.
- **Ownership flag**: `_owns_memory: bool` -> views set this to `false`; `destroy()` skips freeing `_data` for views.
- **No implicit copies**: Transpose, slice, unsqueeze are always O(1) stride/view ops.

### Allocator Strategy

- User provides the allocator at construction time (composition over global state).
- Temporary workspaces (matmul, conv) use a per-thread arena or the user-supplied allocator.
- Future: add a `ScratchAllocator` backed by stack memory for small tensors.

### Error Handling

- Return Zig errors for fallible operations (allocation failure, index bounds, shape mismatch).
- Assert / panic only for programmer errors (`@panic` / `std.debug.assert`).

### Float Precision

- Default compute type: `f32`. `f64` support exists but ops may be slower.
- Document accuracy guarantees: 1 ULP for basic arithmetic, implementation-defined for transcendentals.

### Type Support Matrix

| Type | Signed | Floating | Supported in |
|------|--------|----------|-------------|
| `u8` | No | No | v0.0.x |
| `u32` | No | No | v0.0.x |
| `u64` | No | No | v0.0.x |
| `i32` | Yes | No | v0.0.x |
| `i64` | Yes | No | v0.0.x |
| `f32` | Yes | Yes | v0.0.x |
| `f64` | Yes | Yes | v0.0.x |
| `bool` | -> | -> | v0.0.x |

### Future Considerations (Post-v1.0)

- Sparse tensors (CSR / COO storage)
- GPU / CUDA backend via `std.Target` query or Zig's GPU comptime
- Higher-level `nn.Module`-style API
- `jit` compile of tensor expressions
- Complex number support (`c32`, `c64`)

---

## Testing Strategy

### Per-Function Tests

Every public function must have a corresponding test. Tests should cover:
- Normal path (happy case)
- Edge cases (empty tensor, single element, scalar)
- Error paths (index out of bounds, shape mismatch)
- NaN / inf propagation

### Property-Based Tests

- Arithmetic (commutativity `a+b == b+a`, associativity `(a+b)+c == a+(b+c)`)
- Identity (`a + 0 == a`, `a * 1 == a`)
- Broadcasting (correct output shape)
- Serialization round-trip

### Benchmark Tests

- `zig build bench` -> runs matmul, conv, element-wise benchmarks
- Output: GFLOPS, bandwidth, latency
- Compare against naive baseline to track optimisation progress

### CI Pipeline

- `zig fmt --check` (formatting)
- `zig build test` (all tests, debug mode with safety on)
- `zig build test -Doptimize=ReleaseFast` (tests under optimisation)
- Fuzz step (optional, triggered weekly)

---

## How to Contribute Based on This Plan

1. Pick a version (start with v0.0.x / v0.1.x).
2. Pick an unchecked item within that version.
3. Read the existing code for style (see `STYLE_GUIDE.md`).
4. Implement + test + document in the same PR.
5. Update this plan by checking off the item.

---

*Last updated: 2026-07-07*
