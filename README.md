# Sphūr

**Sphūr (स्फुर्)** is a SIMD™ accelerated PRNG built on top of the
[SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/).

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
