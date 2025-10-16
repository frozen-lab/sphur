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

    #[inline(always)]
    pub fn range_u64<R: IntoSRangeU64>(&self, range: R) -> u64 {
        let (start, span) = range.into_bounds();

        if span == 0 {
            return self.next_u64();
        }

        // rejection sampling to remove bias
        let hi = ((u128::MAX / span as u128) * span as u128) >> 64;
        let zone = ((hi << 64) | 0xFFFF_FFFF_FFFF_FFFF) as u64;

        loop {
            let x = self.next_u64();

            if x < zone {
                return start + Self::mulhi_u64(x, span);
            }
        }
    }

    #[inline(always)]
    pub fn range_u32<R: IntoSRangeU32>(&self, range: R) -> u32 {
        let (start, span) = range.into_bounds();

        if span == 0 {
            return self.next_u32();
        }

        // rejection sampling to remove bias
        let hi = ((u64::MAX / span as u64) * span as u64) >> 32;
        let zone = ((hi << 32) | 0xFFFF_FFFF) as u32;

        loop {
            let x = self.next_u32();

            if x < zone {
                return start + Self::mulhi_u32(x, span);
            }
        }
    }

    #[inline(always)]
    fn mulhi_u64(a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) >> 64) as u64
    }

    #[inline(always)]
    fn mulhi_u32(a: u32, b: u32) -> u32 {
        ((a as u64 * b as u64) >> 32) as u32
    }
}

pub trait IntoSRangeU64 {
    /// Returns (start, span) from a range object where span > 0.
    fn into_bounds(self) -> (u64, u64);
}

impl IntoSRangeU64 for core::ops::Range<u64> {
    fn into_bounds(self) -> (u64, u64) {
        // sanity check
        assert!(self.start < self.end, "gen_range: empty exclusive range");

        let span = self.end - self.start;
        (self.start, span)
    }
}

impl IntoSRangeU64 for core::ops::RangeInclusive<u64> {
    fn into_bounds(self) -> (u64, u64) {
        let start = *self.start();
        let end = *self.end();

        // sanity check
        assert!(start <= end, "gen_range: empty inclusive range");

        // full 64-bit range
        if start == 0 && end == u64::MAX {
            (0, 0)
        } else {
            (start, end - start + 1)
        }
    }
}

pub trait IntoSRangeU32 {
    /// Returns (start, span) from a range object where span > 0.
    fn into_bounds(self) -> (u32, u32);
}

impl IntoSRangeU32 for core::ops::Range<u32> {
    fn into_bounds(self) -> (u32, u32) {
        // sanity check
        assert!(self.start < self.end, "gen_range: empty exclusive range");

        let span = self.end - self.start;
        (self.start, span)
    }
}

impl IntoSRangeU32 for core::ops::RangeInclusive<u32> {
    fn into_bounds(self) -> (u32, u32) {
        let start = *self.start();
        let end = *self.end();

        // sanity check
        assert!(start <= end, "gen_range: empty inclusive range");

        // full 64-bit range
        if start == 0 && end == u32::MAX {
            (0, 0)
        } else {
            (start, end - start + 1)
        }
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

    #[test]
    fn test_range_u64_basic() {
        let rng = Sphur::new_seeded(2025);

        for _ in 0..1000 {
            let v = rng.range_u64(10..20);
            assert!(v >= 10 && v < 20);
        }

        for _ in 0..1000 {
            let v = rng.range_u64(5..=15);
            assert!(v >= 5 && v <= 15);
        }

        let v = rng.range_u64(0..=u64::MAX);
        let _ = v; // should not panic
    }

    #[test]
    fn test_range_u32_basic() {
        let rng = Sphur::new_seeded(909);

        for _ in 0..1000 {
            let v = rng.range_u32(100..200);
            assert!(v >= 100 && v < 200);
        }

        for _ in 0..1000 {
            let v = rng.range_u32(50..=100);
            assert!(v >= 50 && v <= 100);
        }

        let v = rng.range_u32(0..=u32::MAX);
        let _ = v;
    }

    #[test]
    #[should_panic(expected = "empty exclusive range")]
    fn test_empty_range_panics() {
        let rng = Sphur::new_seeded(1);
        let _ = rng.range_u64(10..10);
    }

    #[test]
    fn test_uniformity_rough_check() {
        let rng = Sphur::new_seeded(7777);
        let mut hits = [0u64; 10];

        for _ in 0..10_000 {
            let v = rng.range_u64(0..10);
            hits[v as usize] += 1;
        }

        let avg = hits.iter().sum::<u64>() as f64 / 10.0;
        let max_dev = hits.iter().map(|&x| (x as f64 - avg).abs()).fold(0.0, f64::max);

        assert!(max_dev / avg < 0.25, "rough uniformity check failed");
    }
}
