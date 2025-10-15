# Sphūr

```md
स्पन्दनात् जगत् जायते । तमोनिद्रायाः च महता प्रकाशः ॥
```

Sphūr (स्फुर्) is a SIMD™ accelerated PRNG built on top of the
[SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/).

## Benchmarks

| Function (x86_64 w/ SSSE3) | Time (ns/value)     | Throughput (vals/sec) | CPU (cycles/iter) | IPC  |
|:--------------------------:|:-------------------:|:---------------------:|:-----------------:|:----:|
| next_u64                   |         2.1110 ±0.2 |          473701737.49 | 7.34              | 3.78 |
| next_u32                   |         1.5408 ±0.2 |          649004232.81 | 5.05              | 3.84 |
| batch_u64                  |         0.9997 ±0.2 |         1000273074.55 | -                 | -    |
| batch_u32                  |         0.5030 ±0.2 |         1988111095.65 | -                 | -    |

This benchmarks were conducted on following machine,

* *OS*: WSL2 NixOS 25.05 (Warbler)
* *Kernel*: 6.6.87.2-microsoft-standard-WSL2
* *CPU*: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz
* *Architecture*: x86/64 w/ SSE support

## Randomness

| Metric         | Measured     | Expected   |
|:--------------:|:------------:|:----------:|
| Mean           | 0.500338     | 0.500000   |
| Variance       | 0.083263     | 0.083333   |
| Entropy (bits) | 7.994        | 8.000      |

_📝 NOTE: This results are observed over `1e6` iterations w/ `next_u64` function._

For a **perfect uniform distribution** $X \sim \mathrm{Uniform}(0,1)$ we expect,

| Type                   | Derivation                                                                                                                     |
|:----------------------:|--------------------------------------------------------------------------------------------------------------------------------|
| _Mean (μ)_             | $\( \mu = \frac{a + b}{2} = \frac{0 + 1}{2} \) = \mathbf{0.5}$                                                                 |
| _Variance (σ²)_        | $\( \sigma^2 = \frac{(b - a)^2}{12} = \frac{1^2}{12} \) \approx \mathbf{0.08333}$                                              |
| _Entropy (8-bit bins)_ | $\( H = -\sum_i p_i \log_2 p_i = -256 \cdot \frac{1}{256} \cdot \log_2\!\left(\frac{1}{256}\right) \) \mathbf{8\ \text{bits}}$ |

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

## 🌌 Origin of Śloka

```
स्पन्दनात् जगत् जायते ।
तमोनिद्रायाः च महता प्रकाशः ॥

spandanāt jagat jāyate ।  
tamonidrāyāḥ ca mahatā prakāśaḥ ॥
```

This **Śloka** briefly translates to,

**“From vibration, the universe is born. From the slumber of darkness, awakens the first light.”**

*It is an original Sanskrit composition, inspired by the concept of cosmic vibrations and the Big Bang.*

