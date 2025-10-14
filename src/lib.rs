#![allow(unused)]

mod engine;
mod simd;
mod state;

pub struct Sphur {
    simd: crate::simd::SIMD,
}

impl Sphur {
    #[inline(always)]
    pub fn new_seeded(seed: u64) -> Self {
        Self {
            simd: crate::simd::SIMD::new(seed),
        }
    }

    // #[inline(always)]
    pub fn new() -> Self {
        let seed = crate::simd::platform_seed();
        Sphur::new_seeded(seed)
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.simd.next_u64()
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        self.simd.next_u32()
    }

    #[inline(always)]
    pub fn batch_u64(&mut self, buf: &mut [u64]) {
        self.simd.batch_u64(buf)
    }

    #[inline(always)]
    pub fn batch_u32(&mut self, buf: &mut [u32]) {
        self.simd.batch_u32(buf)
    }
}
