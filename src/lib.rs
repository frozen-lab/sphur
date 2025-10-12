#![allow(unused)]

mod engine;
mod simd;
mod state;

pub struct Sphur {
    simd: crate::simd::SIMD,
}

impl Sphur {
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        Self {
            simd: crate::simd::SIMD::new(seed),
        }
    }

    #[inline(always)]
    pub fn gen_u64(&mut self) -> u64 {
        todo!()
    }
}
