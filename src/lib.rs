#![allow(unused)]

mod engine;
mod simd;
mod state;

pub struct Sphur {
    // NOTE (to compiler): I know what I’m doing. We will create internal mutability through
    // shared refs!
    simd: std::cell::UnsafeCell<crate::simd::SIMD>,
}

/// Impl of `Send`, to allow moving across thread boundries
///
/// NOTE: `Sphur` does not implement `Sync`, as it uses internal mutable state
/// w/o any lock to avoid locking overhead, and regression in perf.
unsafe impl Send for Sphur {}

impl Sphur {
    #[inline(always)]
    pub fn new_seeded(seed: u64) -> Self {
        Self {
            simd: std::cell::UnsafeCell::new(crate::simd::SIMD::new(seed)),
        }
    }

    #[inline(always)]
    pub fn new() -> Self {
        let seed = crate::simd::platform_seed();
        Sphur::new_seeded(seed)
    }

    #[inline(always)]
    pub fn next_u64(&self) -> u64 {
        unsafe { (*self.simd.get()).next_u64() }
    }

    #[inline(always)]
    pub fn next_u32(&self) -> u32 {
        unsafe { (*self.simd.get()).next_u32() }
    }

    #[inline(always)]
    pub fn batch_u64(&self, buf: &mut [u64]) {
        unsafe { (*self.simd.get()).batch_u64(buf) }
    }

    #[inline(always)]
    pub fn batch_u32(&self, buf: &mut [u32]) {
        unsafe { (*self.simd.get()).batch_u32(buf) }
    }
}
