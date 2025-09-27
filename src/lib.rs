//! # Sphūr
//!
//! **Sphūr (स्फुर्)** is a SIMD™ accelerated PRNG built on top of the
//! [SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/).
//!
//! ## Notes
//!
//! - 32-bit targets are **not supported**.
//! - Sphūr is **not cryptographically secure**.
//!
//! ## Platform Support
//!
//! - Linux (x86_64, aarch64)
//! - Mac (x86_64, aarch64)
//! - Windows (x86_64, aarch64)
//!
//! ## Quick Start
//!
//! ```
//! use sphur::Sphur;
//!
//! fn main() {
//!     // auto seed state w/ platform entropy
//!     let mut rng = Sphur::new();
//!
//!     // Generate prng's
//!     let u64_val: u64 = rng.gen_u64();
//!     assert!(u64_val >= 0);
//!
//!     let u32_val: u32 = rng.gen_u32();
//!     assert!(u32_val >= 0);
//!
//!     let ranged_val: u64 = rng.gen_range(10..100);
//!     assert!(ranged_val >= 10);
//!
//!     let flag: bool = rng.gen_bool();
//!     assert!(flag == true || flag == false);
//!
//!     let bounded = rng.gen_range(10..20);
//!     assert!(bounded >= 10 && bounded < 20);
//!
//!     // Reproducible streams with a custom seed
//!     let mut rng1 = Sphur::new_seeded(12345);
//!     let mut rng2 = Sphur::new_seeded(12345);
//!
//!     assert_eq!(rng1.gen_u64(), rng2.gen_u64());
//!
//!     // Bulk generation
//!     let batch = rng.gen_batch();
//!     assert_eq!(batch.len(), 32);
//! }
//! ```

const N_STATE: usize = 16;
const N_U64: usize = N_STATE * 2;

// -----------------------------------------------------------------------------
// Compile guard!
// Sphūr only supports 64-bit architectures (x86_64 and AArch64).
// -----------------------------------------------------------------------------
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("[ERROR]: Sphūr requires 64-bit architecture (x86_64 or AArch64). 32-bit targets (i386/armv7) are not supported.");

/// Sphūr is a SIMD™ accelerated PRNG based on the SFMT algorithm.
///
/// ### Example
///
/// ```
/// use sphur::Sphur;
///
/// let mut rng = Sphur::new_seeded(0x9e3779b97f4a7c15);
///
/// let u64_val: u64 = rng.gen_u64();
/// assert!(u64_val >= 0);
///
/// let u32_val: u32 = rng.gen_u32();
/// assert!(u32_val >= 0);
///
/// let ranged_val: u64 = rng.gen_range(10..100);
/// assert!(ranged_val >= 10);
///
/// let flag: bool = rng.gen_bool();
/// assert!(flag == true || flag == false);
/// ```
#[derive(Clone, Copy)]
pub struct Sphur {
    /// Internal state for SFMT twister
    state: State,

    /// Isa (Instruction Set Architecture) available at runtime
    isa: ISA,

    /// Current idx of the state being used for twister
    idx: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union State {
    #[cfg(target_arch = "x86_64")]
    sse2: [std::arch::x86_64::__m128i; N_STATE],

    #[cfg(target_arch = "x86_64")]
    avx2: [std::arch::x86_64::__m256i; N_STATE / 2],

    #[cfg(target_arch = "aarch64")]
    neon: [core::arch::aarch64::uint64x2_t; N_STATE],
}

impl State {
    #[inline(always)]
    pub unsafe fn as_u64_slice(&self) -> &[u64; N_U64] {
        unsafe { &*(self as *const State as *const [u64; N_U64]) }
    }
}

impl Sphur {
    /// Initialize [Sphur] state from the platform's entropy source.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let val = rng.gen_u64();
    ///
    /// assert!(val >= 0);
    /// ```
    pub fn new() -> Self {
        let seed = Self::platform_seed();
        let isa = ISA::detect_isa();

        Self {
            isa,
            state: Self::init_state(seed, isa),
            idx: 0,
        }
    }

    /// Initialize sphur state w/ an initial seed
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng1 = Sphur::new_seeded(123);
    /// let mut rng2 = Sphur::new_seeded(123);
    ///
    /// assert_eq!(rng1.gen_u64(), rng2.gen_u64());
    /// ```
    pub fn new_seeded(seed: u64) -> Self {
        let isa = ISA::detect_isa();

        Self {
            isa,
            state: Self::init_state(seed, isa),
            idx: 0,
        }
    }

    /// Generate a batch of 32 `u64` values.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let batch: [u64; 32] = rng.gen_batch();
    ///
    /// assert_eq!(batch.len(), 32);
    /// ```
    ///
    /// NOTE: This consumes the entire state (16 * 128-bit).
    pub fn gen_batch(&mut self) -> [u64; N_U64] {
        // Initial twist for a new batch
        self.update_state();
        self.idx = 0;

        unsafe { *self.state.as_u64_slice() }
    }

    /// Generate a random `u64` in the given range (inclusive or exclusive).
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    ///
    /// // Exclusive upper bound
    /// let val1 = rng.gen_range(10..20);
    /// assert!(val1 >= 10 && val1 < 20);
    ///
    /// // Inclusive upper bound
    /// let val2 = rng.gen_range(1..=5);
    /// assert!(val2 >= 1 && val2 <= 5);
    ///
    /// // Single-value inclusive
    /// let val3 = rng.gen_range(42..=42);
    /// assert_eq!(val3, 42);
    /// ```
    pub fn gen_range<R: IntoRangeU64>(&mut self, range: R) -> u64 {
        let (start, span) = range.into_bounds();

        // full 64-bit space: just draw directly
        if span == 0 {
            return self.gen_u64();
        }

        // rejection sampling to remove bias
        let zone = u64::MAX - (u64::MAX % span);

        loop {
            let x = self.gen_u64();

            if x < zone {
                return start + (x % span);
            }
        }
    }

    /// Generate a 64-bit unsigned random nunber.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let val: u64 = rng.gen_u64();
    ///
    /// assert!(val >= 0);
    /// ```
    pub fn gen_u64(&mut self) -> u64 {
        if self.idx >= N_U64 {
            self.update_state();
            self.idx = 0;
        }

        let out = unsafe { self.state.as_u64_slice()[self.idx] };
        self.idx += 1;

        out
    }

    /// Generate a 32-bit unsigned random nunber.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let val: u32 = rng.gen_u32();
    ///
    /// assert!(val >= 0);
    /// ```
    pub fn gen_u32(&mut self) -> u32 {
        (self.gen_u64() & 0xffff_ffff) as u32
    }

    /// Generate a random boolean value.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let b: bool = rng.gen_bool();
    ///
    /// assert!(b == true || b == false);
    /// ```
    #[inline(always)]
    pub fn gen_bool(&mut self) -> bool {
        // just grab 1 random bit
        (self.gen_u64() & 1) != 0
    }

    fn init_state(seed: u64, isa: ISA) -> State {
        let mut tmp = [0u128; N_STATE];
        tmp[0] = seed as u128;

        for i in 1..N_STATE {
            let prev = tmp[i - 1];
            tmp[i] = 6364_1362_2384_6793_005u128
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u128);
        }

        let mut state: Option<State> = None;

        unsafe {
            match isa {
                ISA::AVX2 => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let mut avx_arr = [std::arch::x86_64::_mm256_setzero_si256(); N_STATE / 2];

                        for i in 0..N_STATE / 2 {
                            let lo = tmp[2 * i] as u64;
                            let hi = tmp[2 * i + 1] as u64;
                            let low128 = std::arch::x86_64::_mm_set_epi64x(hi as i64, lo as i64);

                            avx_arr[i] = std::arch::x86_64::_mm256_set_m128i(
                                std::arch::x86_64::_mm_setzero_si128(),
                                low128,
                            );
                        }

                        state = Some(State { avx2: avx_arr });
                    }
                }

                ISA::SSE2 => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let mut arr = [std::arch::x86_64::_mm_setzero_si128(); N_STATE];

                        for i in 0..N_STATE {
                            let lo = tmp[i] as u64;
                            let hi = (tmp[i] >> 64) as u64;

                            arr[i] = std::arch::x86_64::_mm_set_epi64x(hi as i64, lo as i64);
                        }

                        state = Some(State { sse2: arr });
                    }
                }
                ISA::NEON => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        let mut arr = [core::arch::aarch64::vdupq_n_u64(0); N_STATE];

                        for i in 0..N_STATE {
                            let lo = tmp[i] as u64;
                            let hi = (tmp[i] >> 64) as u64;

                            arr[i] = core::arch::aarch64::vsetq_lane_u64(lo, arr[i], 0);
                            arr[i] = core::arch::aarch64::vsetq_lane_u64(hi, arr[i], 1);
                        }

                        state = Some(State { neon: arr });
                    }
                }
            }
        }

        debug_assert!(
            state.is_some(),
            "Error while matching ISA pattern to init the internal state"
        );

        state.unwrap()
    }

    fn update_state(&mut self) {
        unsafe {
            match self.isa {
                ISA::AVX2 => {
                    #[cfg(target_arch = "x86_64")]
                    avx2::twist_block(&mut self.state);
                }
                ISA::SSE2 => {
                    #[cfg(target_arch = "x86_64")]
                    sse2::twist_block(&mut self.state);
                }
                ISA::NEON => {
                    #[cfg(target_arch = "aarch64")]
                    neon::twist_block(&mut self.state);
                }
            }
        }

        self.idx = 0;
    }

    #[cfg(target_arch = "x86_64")]
    fn platform_seed() -> u64 {
        use std::arch::asm;

        unsafe {
            let mut lo: u32;
            let mut hi: u32;

            asm!("rdtsc", out("eax") lo, out("edx") hi);

            ((hi as u64) << 32) | (lo as u64)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn platform_seed() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        // Combine seconds and nanoseconds into a u64
        ((now.as_secs() as u64) << 32) ^ (now.subsec_nanos() as u64)
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use super::{State, N_STATE};
    use std::arch::x86_64::*;

    const POS1: usize = 122 % N_STATE;
    const SL1: i32 = 18;
    const SR1: i32 = 11;
    const SL2: i32 = 1;
    const SR2: i32 = 1;

    #[target_feature(enable = "sse2")]
    pub unsafe fn twist_block(state: &mut State) {
        debug_assert_eq!(std::mem::size_of::<u128>(), std::mem::size_of::<__m128i>());

        unsafe {
            let vecs: &mut [__m128i; N_STATE] = &mut state.sse2;

            for i in 0..N_STATE {
                let x_i = _mm_load_si128(&vecs[i] as *const __m128i);
                let x_pos = _mm_load_si128(&vecs[(i + POS1) % N_STATE] as *const __m128i);
                let x_m1 = _mm_load_si128(&vecs[(i + N_STATE - 1) % N_STATE] as *const __m128i);
                let x_m2 = _mm_load_si128(&vecs[(i + N_STATE - 2) % N_STATE] as *const __m128i);

                let mut y = x_i;

                // y = x_i ^ (x_i << SL1) ^ (x_pos >> SR1) ^ (x_m1 >> SR2) ^ (x_m2 << SL2)
                y = _mm_xor_si128(y, _mm_slli_epi32(x_i, SL1));
                y = _mm_xor_si128(y, _mm_srli_epi32(x_pos, SR1));
                y = _mm_xor_si128(y, _mm_srli_epi32(x_m1, SR2));
                y = _mm_xor_si128(y, _mm_slli_epi32(x_m2, SL2));

                _mm_store_si128(&mut vecs[i] as *mut __m128i, y);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{State, N_STATE};
    use std::arch::x86_64::*;

    const POS1: usize = 122 % N_STATE;
    const SL1: i32 = 18;
    const SR1: i32 = 11;
    const SL2: i32 = 1;
    const SR2: i32 = 1;

    #[target_feature(enable = "avx2")]
    pub unsafe fn twist_block(state: &mut State) {
        unsafe {
            let vecs: &mut [__m256i; N_STATE / 2] = &mut state.avx2;

            for i in 0..(N_STATE / 2) {
                // Get two consecutive 128-bit lanes as one 256-bit
                let x_i = vecs[i];

                // Compute indices for neighboring 128-bit lanes (wrapped)
                let idx_pos = (i + POS1 / 2) % (N_STATE / 2);
                let idx_m1 = (i + (N_STATE / 2) - 1) % (N_STATE / 2);
                let idx_m2 = (i + (N_STATE / 2) - 2) % (N_STATE / 2);

                let x_pos = vecs[idx_pos];
                let x_m1 = vecs[idx_m1];
                let x_m2 = vecs[idx_m2];

                // y = x_i ^ (x_i << SL1) ^ (x_pos >> SR1) ^ (x_m1 >> SR2) ^ (x_m2 << SL2)
                let mut y = x_i;

                y = _mm256_xor_si256(y, _mm256_slli_epi64(x_i, SL1));
                y = _mm256_xor_si256(y, _mm256_srli_epi64(x_pos, SR1));
                y = _mm256_xor_si256(y, _mm256_srli_epi64(x_m1, SR2));
                y = _mm256_xor_si256(y, _mm256_slli_epi64(x_m2, SL2));

                vecs[i] = y;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{State, N_STATE};
    use core::arch::aarch64::*;

    const POS1: usize = 122 % N_STATE;
    const SL1: i32 = 18;
    const SR1: i32 = 11;
    const SL2: i32 = 1;
    const SR2: i32 = 1;

    #[target_feature(enable = "neon")]
    pub unsafe fn twist_block(state: &mut State) {
        unsafe {
            let vecs: &mut [uint64x2_t; N_STATE] = &mut state.neon;

            for i in 0..N_STATE {
                let x_i = vecs[i];
                let x_pos = vecs[(i + POS1) % N_STATE];
                let x_m1 = vecs[(i + N_STATE - 1) % N_STATE];
                let x_m2 = vecs[(i + N_STATE - 2) % N_STATE];

                // y = x_i ^ (x_i << SL1) ^ (x_pos >> SR1) ^ (x_m1 >> SR2) ^ (x_m2 << SL2)
                let mut y = x_i;
                y = veorq_u64(y, vshlq_n_u64(x_i, SL1 as i32));
                y = veorq_u64(y, vshrq_n_u64(x_pos, SR1 as i32));
                y = veorq_u64(y, vshrq_n_u64(x_m1, SR2 as i32));
                y = veorq_u64(y, vshlq_n_u64(x_m2, SL2 as i32));

                vecs[i] = y;
            }
        }
    }
}

#[cfg(test)]
mod simd_tests {
    use super::Sphur;

    #[cfg(target_arch = "aarch64")]
    use super::neon;

    #[cfg(target_arch = "x86_64")]
    use super::{avx2, sse2};

    const TEST_SEED: u64 = 0xBEEF_DEAD_CAFE_BABE;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse2_twist_block_deterministic() {
        let mut state1 = Sphur::init_state(TEST_SEED, crate::ISA::SSE2);
        let mut state2 = Sphur::init_state(TEST_SEED, crate::ISA::SSE2);

        unsafe {
            sse2::twist_block(&mut state1);
            sse2::twist_block(&mut state2);

            let arr1 = state1.as_u64_slice();
            let arr2 = state2.as_u64_slice();

            assert_eq!(arr1, arr2, "SSE2 twist_block should be deterministic");

            for &v in arr1 {
                assert_ne!(v, 0, "SSE2 twist_block should update all lanes");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_twist_block_deterministic() {
        let mut state1 = Sphur::init_state(TEST_SEED, crate::ISA::AVX2);
        let mut state2 = Sphur::init_state(TEST_SEED, crate::ISA::AVX2);

        unsafe {
            avx2::twist_block(&mut state1);
            avx2::twist_block(&mut state2);

            let arr1 = state1.as_u64_slice();
            let arr2 = state2.as_u64_slice();

            assert_eq!(arr1, arr2, "AVX2 twist_block should be deterministic");

            for &v in arr1 {
                assert_ne!(v, 0, "AVX2 twist_block should update all lanes");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_twist_block_deterministic() {
        let mut state1 = Sphur::init_state(TEST_SEED, crate::ISA::NEON);
        let mut state2 = Sphur::init_state(TEST_SEED, crate::ISA::NEON);

        unsafe {
            neon::twist_block(&mut state1);
            neon::twist_block(&mut state2);

            let arr1 = state1.as_u64_slice();
            let arr2 = state2.as_u64_slice();

            assert_eq!(arr1, arr2, "NEON twist_block should be deterministic");

            for &v in arr1 {
                assert_ne!(v, 0, "NEON twist_block should update all lanes");
            }
        }
    }
}

pub trait IntoRangeU64 {
    /// Returns (start, span) from a range object where span > 0.
    fn into_bounds(self) -> (u64, u64);
}

impl IntoRangeU64 for core::ops::Range<u64> {
    fn into_bounds(self) -> (u64, u64) {
        assert!(self.start < self.end, "gen_range: empty exclusive range");

        let span = self.end - self.start;
        (self.start, span)
    }
}

impl IntoRangeU64 for core::ops::RangeInclusive<u64> {
    fn into_bounds(self) -> (u64, u64) {
        let start = *self.start();
        let end = *self.end();

        assert!(start <= end, "gen_range: empty inclusive range");

        // full 64-bit range
        if start == 0 && end == u64::MAX {
            (0, 0)
        } else {
            (start, end - start + 1)
        }
    }
}

#[cfg(test)]
mod into_range_u64_tests {
    use super::*;

    const TEST_SEED: u64 = 0xBEEF_DEAD_CAFE_BABE;

    #[test]
    fn test_gen_range_inclusive() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        for _ in 0..1000 {
            let x = rng.gen_range(1..=5);

            assert!(
                (1..=5).contains(&x),
                "gen_range inclusive returned out-of-range value: {}",
                x
            );
        }
    }

    #[test]
    fn test_gen_range_single_value() {
        let mut rng = Sphur::new_seeded(TEST_SEED);
        let x = rng.gen_range(42..=42);

        assert_eq!(
            x, 42,
            "Single-value inclusive range should return the value itself"
        );
    }

    #[test]
    fn test_gen_range_full_span() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        // Generate multiple values in 0..=u64::MAX
        for _ in 0..1_000 {
            let x = rng.gen_range(0..=u64::MAX);

            // Assert that x is within the valid u64 range
            // (This should always be true)
            assert!(x <= u64::MAX, "Value exceeded u64::MAX: {}", x);
        }
    }

    #[test]
    fn test_gen_range_inclusive_small() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        // Example of small inclusive range
        for _ in 0..1_000 {
            let x = rng.gen_range(1..=5);
            assert!((1..=5).contains(&x), "Value out of range: {}", x);
        }
    }

    #[test]
    fn test_gen_range_exclusive_small() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        // Example of small exclusive range
        for _ in 0..1_000 {
            let x = rng.gen_range(10..20);
            assert!((10..20).contains(&x), "Value out of range: {}", x);
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(unused)]
enum ISA {
    // This SIMD ISA is an upgrade over SSE2 if available at runtime
    AVX2,

    // This SIMD ISA is default on x64 (x86_64), as it's virtually
    // available on all x64 CPU's
    SSE2,

    // Neon is vurtually available on all aarch64 CPU's
    NEON,
}

impl ISA {
    #[cfg(target_arch = "x86_64")]
    fn detect_isa() -> ISA {
        if is_x86_feature_detected!("avx2") {
            return ISA::AVX2;
        }

        ISA::SSE2
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_isa() -> ISA {
        ISA::NEON
    }
}

#[cfg(test)]
mod isa_tests {
    use super::ISA;

    #[test]
    fn test_detect_isa_is_correct() {
        let isa = ISA::detect_isa();

        #[cfg(target_arch = "x86_64")]
        match isa {
            ISA::AVX2 | ISA::SSE2 => {}
            _ => panic!("Unknown ISA detected for x86_64"),
        }

        #[cfg(target_arch = "aarch64")]
        match isa {
            ISA::NEON => {}
            _ => panic!("Unknown ISA detected for aarch64"),
        }
    }
}

// #[cfg(test)]
// mod init_tests {
//     use super::*;

//     #[test]
//     fn test_platform_seed_non_zero() {
//         let seed = Sphur::platform_seed();
//         assert_ne!(seed, 0, "Platform seed should not be zero");
//     }

//     #[test]
//     fn test_platform_seed_changes() {
//         let seed1 = Sphur::platform_seed();

//         // tiny delay
//         std::thread::sleep(std::time::Duration::from_micros(1));

//         let seed2 = Sphur::platform_seed();

//         assert_ne!(
//             seed1, seed2,
//             "Two calls to platform_seed should differ over time"
//         );
//     }

//     #[test]
//     fn test_new_initializes_state() {
//         let sphur = Sphur::new();

//         assert!(
//             sphur.state.0.iter().any(|&x| x != 0),
//             "State should not be all zero"
//         );

//         assert_eq!(sphur.idx, 0);
//     }

//     #[test]
//     fn test_new_different_seeds_generate_different_states() {
//         let sphur1 = Sphur::new();

//         // Slight delay ensures platform_seed changes
//         std::thread::sleep(std::time::Duration::from_micros(1));

//         let sphur2 = Sphur::new();

//         assert_ne!(
//             sphur1.state, sphur2.state,
//             "Different platform seeds should generate different states"
//         );
//     }

//     #[test]
//     fn test_state_is_deterministic() {
//         let seed = 123456789u64;
//         let sphur1 = Sphur::new_seeded(seed);
//         let sphur2 = Sphur::new_seeded(seed);

//         assert_eq!(
//             sphur1.state, sphur2.state,
//             "State should be deterministic for same seed"
//         );
//     }

//     #[test]
//     fn test_state_differs_for_different_seeds() {
//         let sphur1 = Sphur::new_seeded(1u64);
//         let sphur2 = Sphur::new_seeded(2u64);

//         assert_ne!(
//             sphur1.state, sphur2.state,
//             "State should differ for different seeds"
//         );
//     }

//     #[test]
//     fn test_init_state_matches_new_seeded() {
//         let seed = 2025u64;
//         let state_from_init = Sphur::init_state(seed);
//         let sphur = Sphur::new_seeded(seed);

//         assert_eq!(
//             state_from_init, sphur.state,
//             "init_state should match new_seeded output"
//         );
//     }
// }

#[cfg(test)]
mod rand_gen_tests {
    use super::*;

    const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

    #[test]
    fn test_gen_u64_uniformity_sample() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        let mut counts = [0u32; 10];
        let n = 100_000;

        for _ in 0..n {
            let val = rng.gen_range(0..10);

            counts[val as usize] += 1;
        }

        // Expect roughly uniform distribution across bins
        let mean = n as f64 / 10.0;

        for &c in &counts {
            assert!(
                (0.9 * mean..1.1 * mean).contains(&(c as f64)),
                "Bin count {} deviates more than 10% from mean {}",
                c,
                mean
            );
        }
    }

    #[test]
    fn test_gen_batch_consistency() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        let batch1 = rng.gen_batch();
        let batch2 = rng.gen_batch();

        // After a full batch, state should update and produce different output
        assert_ne!(batch1, batch2, "Consecutive batches should differ");
    }

    #[test]
    fn test_gen_batch_length() {
        let mut rng = Sphur::new_seeded(TEST_SEED);
        let batch = rng.gen_batch();

        assert_eq!(batch.len(), N_U64, "Batch length should match N_U64");
    }

    #[test]
    fn test_deterministic_u64() {
        let mut rng1 = Sphur::new_seeded(TEST_SEED);
        let mut rng2 = Sphur::new_seeded(TEST_SEED);

        for _ in 0..100 {
            assert_eq!(
                rng1.gen_u64(),
                rng2.gen_u64(),
                "gen_u64 should be deterministic for same seed"
            );
        }
    }

    #[test]
    fn test_gen_u32_lanes() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        let mut seen = [0u32; 4];

        for i in 0..4 {
            seen[i] = rng.gen_u32();
        }

        assert!(
            seen.iter().any(|&v| v != 0),
            "at least one lane should be non-zero"
        );
    }

    #[test]
    fn test_gen_range_exclusive() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        for _ in 0..1000 {
            let x = rng.gen_range(10..20);

            assert!(
                x >= 10 && x < 20,
                "gen_range exclusive returned out-of-range value: {}",
                x
            );
        }
    }

    #[test]
    fn test_gen_range_inclusive() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        for _ in 0..1000 {
            let x = rng.gen_range(1..=5);

            assert!(
                x >= 1 && x <= 5,
                "gen_range inclusive returned out-of-range value: {}",
                x
            );
        }
    }

    #[test]
    fn test_gen_range_full_span() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        // Generate multiple values in 0..=u64::MAX to ensure no panic
        for _ in 0..1000 {
            let x = rng.gen_range(0..=u64::MAX);

            assert!(x <= u64::MAX, "Value exceeded u64::MAX: {}", x);
        }
    }

    #[test]
    fn test_gen_bool_distribution() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        let mut trues = 0;
        let mut falses = 0;

        for _ in 0..10_000 {
            if rng.gen_bool() {
                trues += 1;
            } else {
                falses += 1;
            }
        }

        let ratio = trues as f64 / (trues + falses) as f64;
        assert!(
            (0.45..0.55).contains(&ratio),
            "gen_bool distribution skewed: {} trues / {} falses = {}",
            trues,
            falses,
            ratio
        );
    }
}
