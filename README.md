# Sphūr

**Sphūr (स्फुर्)** is a SIMD™ accelerated PRNG built on top of the
[SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/).

## Benchmarks

| Function (x86_64 w/ SSE) | Time (ns/value)     | Throughput (vals/sec) | CPU (cycle/iter) | IPC  | GHz  |
|:------------------------:|:-------------------:|:---------------------:|:----------------:|:----:|:----:|
| next_u64                 |         3.8234 ±0.2 |          249160515.93 | 13.38            | 2.97 | 4.25 |
| next_u32                 |         1.7220 ±0.2 |          588826371.88 | 17.38            | 3.02 | 4.14 |
| batch_64                 |         1.7096 ±0.2 |          611326909.51 | -                | -    | -    |
| batch_u32                |         0.7443 ±0.2 |         1251766555.55 | -                | -    | -    |

## Randomness

For a **perfect uniform distribution** `X ~ Uniform(0, 1)` we expect,

**Mean (μ):** μ = (a + b) / 2 = (0 + 1) / 2 = **0.5**

**Variance (σ²):** σ² = (b − a)² / 12 = (1)² / 12 ≈ **0.08333**  
*(theoretical variance of any Uniform[0,1) RNG)*

**Entropy:**  H = −∑ pᵢ log₂ pᵢ = −256 × (1/256) × log₂(1/256) = **8 bits**  
For 8-bit bins (256 equally likely values):  
*(i.e., 8 bits of uncertainty per byte)*

**Observed Results (1 Million samples from `next_u64`)**

| Metric     | Measured     | Expected   |
|:----------:|:------------:|:----------:|
| Mean       | 0.500338     | 0.500000   |
| Variance   | 0.083263     | 0.083333   |
| Entropy    | 7.994 bits   | 8.000 bits |

## Thread Safety

`Sphur` is **`Send` but not `Sync`**.
Which implies you can safely **move it across threads**, but can't **share** the same instance
between threads concurrently.

This design keeps the generator **lock-free w/ zero-overhead**, to facilitate good SIMD perf.

For parallel workloads, you can simply create one `Sphur` instance per thread internally, each will
maintain its own independent state, guaranteeing both **thread safety** and **zero overhead**.

## Internal Mutability

`Sphur` uses **interior mutability** via `UnsafeCell`, allowing mutation of its internal state through
shared references.

We avoid mutex and thread locks to keep operations uninterrupted and fast 🔥🚀!

But this comes at the cost, `Sphur` doesn’t implement `Sync`, so it can’t be shared across threads.
