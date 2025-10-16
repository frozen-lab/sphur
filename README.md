![Tests](https://github.com/frozen-lab/sphur/actions/workflows/tests.yaml/badge.svg)
![Release](https://github.com/frozen-lab/sphur/actions/workflows/release.yaml/badge.svg)
![Documentation](https://docs.rs/sphur/badge.svg)
![Downloads](https://img.shields.io/crates/d/sphur.svg)
![Crates.io](https://img.shields.io/crates/v/sphur.svg)
![License](https://img.shields.io/github/license/frozen-lab/sphur)

![Linux x86_64](https://img.shields.io/badge/x86_64-linux-black)
![Linux aarch64](https://img.shields.io/badge/aarch64-linux-black)
![Windows x86_64](https://img.shields.io/badge/x86_64-windows-black)
![Windows aarch64](https://img.shields.io/badge/aarch64-windows-black)
![Mac x86_64](https://img.shields.io/badge/x86_64-macos-black)
![Mac aarch64](https://img.shields.io/badge/aarch64-macos-black)

# Sphūr

```md
स्पन्दनात् जगत् जायते । तमोनिद्रायाः च महता प्रकाशः ॥
```

Sphūr (स्फुर्) is a SIMD™ accelerated PRNG built on top of the
[SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/).

## Benchmarks (x86_64)

| Function     | Time (ns/value)     | Throughput (vals/sec) | CPU (cycles/iter) | IPC  |
|:------------:|:-------------------:|:---------------------:|:-----------------:|:----:|
| next_u64     |         2.1110 ±0.2 |          473701737.49 | 7.34              | 3.78 |
| next_u32     |         1.5408 ±0.2 |          649004232.81 | 5.05              | 3.84 |
| batch_u64    |         0.9997 ±0.2 |         1000273074.55 | -                 | -    |
| batch_u32    |         0.5030 ±0.2 |         1988111095.65 | -                 | -    |

This benchmarks were conducted on following machine,

* *OS*: WSL2 NixOS 25.05 (Warbler)
* *Kernel*: 6.6.87.2-microsoft-standard-WSL2
* *CPU*: Intel Core i5-10300H @ 2.50GHz
* *Architecture*: x86/64 w/ SSE

## Benchmakrs (aarch64)

| Function     | Time (ns/value)     | Throughput (vals/sec) |
|:------------:|:-------------------:|:---------------------:|
| next_u64     |         2.6978 ±0.2 |          370665529.96 |
| next_u32     |         2.1539 ±0.2 |          464276047.40 |
| batch_u64    |         1.7329 ±0.2 |          577078333.19 |
| batch_u32    |         0.8755 ±0.2 |         1142201193.03 |

This benchmarks were conducted on following machine,

* *OS*: Ubuntu 25.10 (GCP)
* *Kernel*: 6.17.0-1002-gcp
* *CPU*: Neoverse-V2 (Google Axion)
* *Architecture*: aarch64 w/ NEON

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

