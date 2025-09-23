# Sphūr

[![Unit Tests](https://github.com/frozen-lab/sphur/actions/workflows/unit_tests.yaml/badge.svg?branch=master)](https://github.com/frozen-lab/sphur/actions/workflows/unit_tests.yaml)
[![Release](https://github.com/frozen-lab/sphur/actions/workflows/release.yaml/badge.svg)](https://github.com/frozen-lab/sphur/actions/workflows/release.yaml)
[![Crates.io](https://img.shields.io/crates/v/sphur.svg)](https://crates.io/crates/sphur)
[![Documentation](https://docs.rs/sphur/badge.svg)](https://docs.rs/sphur)
[![Rust](https://img.shields.io/badge/rust-1.89.0%2B-blue.svg?maxAge=3600)](https://github.com/frozen-lab/sphur/)
[![Pull Requests](https://img.shields.io/github/issues-pr/frozen-lab/sphur?logo=github&logoColor=white)](https://github.com/frozen-lab/sphur/pulls)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/frozen-lab/sphur?logo=github&logoColor=white)](https://github.com/frozen-lab/sphur/issues)
[![License](https://img.shields.io/github/license/frozen-lab/sphur?logo=open-source-initiative&logoColor=white)](https://github.com/frozen-lab/sphur/blob/master/LICENSE)

**Sphūr (स्फुर्)** is a SIMD™ accelerated PRNG built on top of the
[SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/).

> [!WARNING]
> 32-bit targets are **not supported**.

> [!IMPORTANT]
> Sphūr is **not cryptographically secure**.  

## Platform Support

- ✅ Linux (x86_64, aarch64)
- ✅ macOS (x86_64, aarch64)
- ✅ Windows (x86_64, aarch64)

## Benchmarks

| API            | Throughput (numbers/µs) |
|:--------------:|:-----------------------:|
| gen_u128       |                  314.46 |
| gen_u64        |                  393.92 |
| gen_u32        |                  413.33 |
| gen_bool       |                  373.69 |
| gen_batch (32) |                  170.49 |

According to the benchmakrs, **Sphūr** generates about _393.92_ `u64` numbers/µs,
which is good enough for normal usage. It means you can generate about ~394 million
`u64` numbers per second using AVX2!

| Metric                    | Value       |
|:-------------------------:|:-----------:|
| Chi-squared (uniformity)  |     231.60  |
| Autocorrelation (lag 1)   |   -0.01556  |

- The **Chi-Squared** represents uniformity. It represents how evenly distributed
the generated numbers are across 256 bins. The prefect uniform distribution
is `NUM_BINS - 1 = 255`. Sphūr's 231.60 is very close to the expected value and well
within normal statistical variation.

- **Autocorrelation** represents the correlation between consecutive random numbers.
Ideally, consecutive numbers should be independent, giving a value close to 0.
Sphūr's `-0.01556` is very low, indicating almost no correlation between successive numbers.

<div align="center">
 <figure>
   <img src="https://raw.githubusercontent.com/frozen-lab/sphur/refs/heads/master/assets/dist_plot.png" alt="Distribution Plot" />
   <figcaption>Distribution Plot of 1000 (u64) PRN's</figcaption>
 </figure>
</div>

## Quick Start

Read [Sphur](https://docs.rs/sphur/latest/sphur/struct.Sphur.html) docs for usgae guide.

```rs
use sphur::Sphur;

fn main() {
   // auto seed state w/ platform entropy
   let mut rng = Sphur::new();

   // Generate prng's
   let x128: u128 = rng.gen_u128();
   let x64: u64 = rng.gen_u64();
   let x32: u32 = rng.gen_u32();
   let flag: bool = rng.gen_bool();

   let bounded = rng.gen_range(10..20);
   assert!(bounded >= 10 && bounded < 20);

   // Reproducible streams with a custom seed
   let mut rng1 = Sphur::new_seeded(12345);
   let mut rng2 = Sphur::new_seeded(12345);

   assert_eq!(rng1.gen_u64(), rng2.gen_u64());

   // Bulk generation
   let batch = rng.gen_batch();
   assert_eq!(batch.len(), 32);
}
```

Refer to [docs.rs](https://docs.rs/sphur/latest/sphur/) for detailed documentation.
