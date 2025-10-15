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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_for_same_seed() {
        let a = Sphur::new_seeded(123456789);
        let b = Sphur::new_seeded(123456789);

        let mut seq_a = [0u64; 8];
        let mut seq_b = [0u64; 8];

        for i in 0..8 {
            seq_a[i] = a.next_u64();
            seq_b[i] = b.next_u64();
        }

        assert_eq!(seq_a, seq_b, "identical seeds must yield same sequence");
    }

    #[test]
    fn test_different_seeds_produce_different_sequences() {
        let a = Sphur::new_seeded(1);
        let b = Sphur::new_seeded(2);

        let mut seq_a = [0u64; 8];
        let mut seq_b = [0u64; 8];

        for i in 0..8 {
            seq_a[i] = a.next_u64();
            seq_b[i] = b.next_u64();
        }

        assert_ne!(seq_a, seq_b, "different seeds should yield distinct output");
    }

    #[test]
    fn test_next_u32_and_u64_work_consistently() {
        let rng = Sphur::new_seeded(999);

        let _ = rng.next_u32();
        let _ = rng.next_u64();
    }

    #[test]
    fn test_batch_rngerates_full_buffer() {
        let rng = Sphur::new_seeded(42);

        let mut buf64 = [0u64; 32];
        let mut buf32 = [0u32; 32];

        rng.batch_u64(&mut buf64);
        rng.batch_u32(&mut buf32);

        assert!(buf64.iter().any(|&x| x != 0), "batch_u64 must fill buffer");
        assert!(buf32.iter().any(|&x| x != 0), "batch_u32 must fill buffer");
    }

    #[test]
    fn test_can_move_across_threads_but_not_shared() {
        use std::thread;

        let rng = Sphur::new_seeded(1234);
        let handle = thread::spawn(move || {
            let mut out = [0u64; 4];

            for i in 0..4 {
                out[i] = rng.next_u64();
            }

            out
        });

        let res = handle.join().expect("thread should run successfully");
        assert!(res.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_repeated_batch_and_next_consistent() {
        let mut buf = [0u64; 4];
        let mut seq = [0u64; 4];

        let rng = Sphur::new_seeded(777);
        rng.batch_u64(&mut buf);

        for i in 0..4 {
            seq[i] = rng.next_u64();
        }

        assert!(buf.iter().any(|&x| x != seq[0]));
    }
}
