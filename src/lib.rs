//! # Sphūr
//!
//! **Sphūr** is a SIMD™ accelerated PRNG built on top of the
//! [SFMT (SIMD-oriented Fast Mersenne Twister)](https://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/).
//!
//! ## Platform Support
//!
//! - ✅ Linux (x86_64, aarch64)
//! - ✅ macOS (x86_64, aarch64)
//! - ✅ Windows (x86_64, aarch64)
//!
//! **WARN:** 32-bit targets are **not supported**.
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
//!     let x128: u128 = rng.gen_u128();
//!     let x64: u64 = rng.gen_u64();
//!     let x32: u32 = rng.gen_u32();
//!     let flag: bool = rng.gen_bool();
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
//!
//! **NOTE:** Sphūr is **not cryptographically secure**.  
//!

const N_STATE: usize = 16;

// -----------------------------------------------------------------------------
// Architecture guard!
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
/// let val: u128 = rng.gen_u128();
/// assert!(val >= 0);
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
#[derive(Debug, Clone, Copy)]
pub struct Sphur {
    /// Internal state for SFMT twister
    state: State,

    /// Isa (Instruction Set Architecture) available at runtime
    isa: ISA,

    /// Current idx of the state being used for twister
    idx: usize,
}

#[repr(align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
struct State([u128; N_STATE]);

impl Sphur {
    /// Initialize [Sphur] state from the platform's entropy source.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let val = rng.gen_u128();
    ///
    /// assert!(val >= 0);
    /// ```
    pub fn new() -> Self {
        let seed = Self::platform_seed();

        Self {
            state: Self::init_state(seed),
            isa: ISA::detect_isa(),
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
        Self {
            state: Self::init_state(seed),
            isa: ISA::detect_isa(),
            idx: 0,
        }
    }

    /// Generate a 128-bit unsigned random number.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let val: u128 = rng.gen_u128();
    ///
    /// assert!(val > 0);
    /// ```
    pub fn gen_u128(&mut self) -> u128 {
        if self.idx >= N_STATE {
            self.update_state();
        }

        let v = self.state.0[self.idx];
        self.idx += 1;

        v
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
    pub fn gen_batch(&mut self) -> [u64; N_STATE * 2] {
        // Initial twist for a new batch
        self.update_state();

        let mut out = [0u64; N_STATE * 2];

        for (i, word) in self.state.0.iter().enumerate() {
            out[2 * i] = (*word & 0xffff_ffff_ffff_ffff) as u64;
            out[2 * i + 1] = (word >> 64) as u64;
        }

        out
    }

    /// Generate a random `u64` in the exclusive range `[start, end)`.
    ///
    /// NOTE: Includes assertion for `start >= end`.
    ///
    /// ### Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new();
    /// let val = rng.gen_range(10..20);
    ///
    /// assert!(val >= 10 && val < 20);
    /// ```
    pub fn gen_range(&mut self, range: core::ops::Range<u64>) -> u64 {
        assert!(range.start < range.end, "gen_range: empty range");

        let span = range.end - range.start;

        // full 64-bit space
        if span == 0 {
            return self.gen_u64();
        }

        let zone = u64::MAX - (u64::MAX % span);

        loop {
            let x = self.gen_u64();

            if x < zone {
                return range.start + (x % span);
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
        if self.idx >= N_STATE * 2 {
            self.update_state();
        }

        let word = self.state.0[self.idx >> 1]; // idx/2
        let hi = (word >> 64) as u64;
        let lo = word as u64;

        // mask = 0 if even, !0 if odd
        let mask = -((self.idx & 1) as i64) as u64;

        let out = (lo & !mask) | (hi & mask);

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
        if self.idx >= N_STATE * 4 {
            self.update_state();
            self.idx = 0;
        }

        // idx / 4 -> u128
        let word = self.state.0[self.idx >> 2];

        // 4 u32 lanes from u128
        let w0 = word as u32;
        let w1 = (word >> 32) as u32;
        let w2 = (word >> 64) as u32;
        let w3 = (word >> 96) as u32;

        let lane = (self.idx & 3) as u32;

        // mask selection
        let m0 = ((lane == 0) as u32).wrapping_neg();
        let m1 = ((lane == 1) as u32).wrapping_neg();
        let m2 = ((lane == 2) as u32).wrapping_neg();
        let m3 = ((lane == 3) as u32).wrapping_neg();

        let out = (w0 & m0) | (w1 & m1) | (w2 & m2) | (w3 & m3);
        self.idx += 1;

        out
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

    fn init_state(seed: u64) -> State {
        let mut state = [0u128; 16];

        // initial seed
        state[0] = seed as u128;

        for i in 1..N_STATE {
            let prev = state[i - 1];

            state[i] = 6364136223846793005u128
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u128);
        }

        State(state)
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
    pub fn twist_block(state: &mut State) {
        // sanity check
        debug_assert_eq!(std::mem::size_of::<u128>(), std::mem::size_of::<__m128i>());

        unsafe {
            let ptr = state.0.as_mut_ptr() as *mut __m128i;
            let vecs: &mut [__m128i; N_STATE] = &mut *(ptr as *mut [__m128i; N_STATE]);
            let mut i = 0;

            while i < N_STATE {
                // X[i], X[i+POS1], X[i-1], X[i-2]
                let x_i = _mm_load_si128(&vecs[i] as *const __m128i);
                let x_pos = _mm_load_si128(&vecs[(i + POS1) % N_STATE] as *const __m128i);
                let x_m1 = _mm_load_si128(&vecs[(i + N_STATE - 1) % N_STATE] as *const __m128i);
                let x_m2 = _mm_load_si128(&vecs[(i + N_STATE - 2) % N_STATE] as *const __m128i);

                let mut y = x_i;

                y = _mm_xor_si128(y, _mm_slli_epi32(x_i, SL1)); // shift left logical by SL1 bits
                y = _mm_xor_si128(y, _mm_srli_epi32(x_pos, SR1)); // shift right logical by SR1 bits
                y = _mm_xor_si128(y, _mm_srli_epi32(x_m1, SR2));
                y = _mm_xor_si128(y, _mm_slli_epi32(x_m2, SL2));

                // update the state
                _mm_store_si128(&mut vecs[i] as *mut __m128i, y);

                i += 1;
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
        debug_assert_eq!(std::mem::size_of::<u128>(), 16);

        unsafe {
            for i in 0..N_STATE {
                let x_i_ptr = state.0.as_ptr().add(i) as *const __m128i;
                let x_i = _mm_loadu_si128(x_i_ptr);

                let idx_pos = (i + POS1) % N_STATE;
                let x_pos_ptr = state.0.as_ptr().add(idx_pos) as *const __m128i;
                let x_pos = _mm_loadu_si128(x_pos_ptr);

                let idx_m1 = (i + N_STATE - 1) % N_STATE;
                let x_m1_ptr = state.0.as_ptr().add(idx_m1) as *const __m128i;
                let x_m1 = _mm_loadu_si128(x_m1_ptr);

                let idx_m2 = (i + N_STATE - 2) % N_STATE;
                let x_m2_ptr = state.0.as_ptr().add(idx_m2) as *const __m128i;
                let x_m2 = _mm_loadu_si128(x_m2_ptr);

                let mut y = _mm256_set_m128i(_mm_setzero_si128(), x_i); // upper 128 zero, lower x_i

                y = _mm256_xor_si256(y, _mm256_slli_epi64(y, SL1));
                y = _mm256_xor_si256(
                    y,
                    _mm256_srli_epi64(_mm256_set_m128i(_mm_setzero_si128(), x_pos), SR1),
                );
                y = _mm256_xor_si256(
                    y,
                    _mm256_srli_epi64(_mm256_set_m128i(_mm_setzero_si128(), x_m1), SR2),
                );
                y = _mm256_xor_si256(
                    y,
                    _mm256_slli_epi64(_mm256_set_m128i(_mm_setzero_si128(), x_m2), SL2),
                );

                // extract lower 128-bit
                let y_lo = _mm256_castsi256_si128(y);
                let out_ptr = state.0.as_mut_ptr().add(i) as *mut __m128i;

                // store back the state
                _mm_storeu_si128(out_ptr, y_lo);
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
        for i in 0..N_STATE {
            let x_i_ptr = state.0.as_ptr().add(i) as *const u64;
            let x_i = vld1q_u64(x_i_ptr);

            let idx_pos = (i + POS1) % N_STATE;
            let x_pos_ptr = state.0.as_ptr().add(idx_pos) as *const u64;
            let x_pos = vld1q_u64(x_pos_ptr);

            let idx_m1 = (i + N_STATE - 1) % N_STATE;
            let x_m1_ptr = state.0.as_ptr().add(idx_m1) as *const u64;
            let x_m1 = vld1q_u64(x_m1_ptr);

            let idx_m2 = (i + N_STATE - 2) % N_STATE;
            let x_m2_ptr = state.0.as_ptr().add(idx_m2) as *const u64;
            let x_m2 = vld1q_u64(x_m2_ptr);

            // y = x_i ^ (x_i << SL1) ^ (x_pos >> SR1) ^ (x_m1 >> SR2) ^ (x_m2 << SL2)
            let mut y = x_i;

            y = veorq_u64(y, vshlq_n_u64(x_i, SL1 as i32));
            y = veorq_u64(y, vshrq_n_u64(x_pos, SR1 as i32));
            y = veorq_u64(y, vshrq_n_u64(x_m1, SR2 as i32));
            y = veorq_u64(y, vshlq_n_u64(x_m2, SL2 as i32));

            let out_ptr = state.0.as_mut_ptr().add(i) as *mut u64;

            // store the state back
            vst1q_u64(out_ptr, y);
        }
    }
}

#[cfg(test)]
mod simd_tests {
    #[cfg(target_arch = "aarch64")]
    use super::neon;

    #[cfg(target_arch = "x86_64")]
    use super::{avx2, sse2};

    use super::{State, N_STATE};

    fn init_test_state() -> State {
        let mut state = [0u128; N_STATE];

        for i in 0..N_STATE {
            state[i] = i as u128 * 0xDEADBEEFCAFEBABE;
        }

        State(state)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse2_twist_block_deterministic() {
        let mut state1 = init_test_state();
        let mut state2 = init_test_state();

        unsafe {
            sse2::twist_block(&mut state1);
            sse2::twist_block(&mut state2);
        }

        assert_eq!(
            state1.0, state2.0,
            "SSE2 twist_block should be deterministic"
        );

        for &v in &state1.0 {
            assert_ne!(v, 0, "SSE2 twist_block should update all lanes");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_twist_block_deterministic() {
        let mut state1 = init_test_state();
        let mut state2 = init_test_state();

        unsafe {
            avx2::twist_block(&mut state1);
            avx2::twist_block(&mut state2);
        }

        assert_eq!(
            state1.0, state2.0,
            "AVX2 twist_block should be deterministic"
        );

        for &v in &state1.0 {
            assert_ne!(v, 0, "AVX2 twist_block should update all lanes");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_twist_block_deterministic() {
        let mut state1 = init_test_state();
        let mut state2 = init_test_state();

        unsafe {
            neon::twist_block(&mut state1);
            neon::twist_block(&mut state2);
        }

        assert_eq!(
            state1.0, state2.0,
            "NEON twist_block should be deterministic"
        );

        for &v in &state1.0 {
            assert_ne!(v, 0, "NEON twist_block should update all lanes");
        }
    }
}

#[cfg(test)]
mod init_tests {
    use super::*;

    #[test]
    fn test_platform_seed_non_zero() {
        let seed = Sphur::platform_seed();
        assert_ne!(seed, 0, "Platform seed should not be zero");
    }

    #[test]
    fn test_platform_seed_changes() {
        let seed1 = Sphur::platform_seed();

        // tiny delay
        std::thread::sleep(std::time::Duration::from_micros(1));

        let seed2 = Sphur::platform_seed();

        assert_ne!(
            seed1, seed2,
            "Two calls to platform_seed should differ over time"
        );
    }

    #[test]
    fn test_new_initializes_state() {
        let sphur = Sphur::new();

        assert!(
            sphur.state.0.iter().any(|&x| x != 0),
            "State should not be all zero"
        );

        assert_eq!(sphur.idx, 0);
    }

    #[test]
    fn test_new_different_seeds_generate_different_states() {
        let sphur1 = Sphur::new();

        // Slight delay ensures platform_seed changes
        std::thread::sleep(std::time::Duration::from_micros(1));

        let sphur2 = Sphur::new();

        assert_ne!(
            sphur1.state, sphur2.state,
            "Different platform seeds should generate different states"
        );
    }

    #[test]
    fn test_state_first_element_matches_seed() {
        let seed = 42u64;
        let sphur = Sphur::new_seeded(seed);

        assert_eq!(sphur.state.0[0], seed as u128);
    }

    #[test]
    fn test_state_is_deterministic() {
        let seed = 123456789u64;
        let sphur1 = Sphur::new_seeded(seed);
        let sphur2 = Sphur::new_seeded(seed);

        assert_eq!(
            sphur1.state, sphur2.state,
            "State should be deterministic for same seed"
        );
    }

    #[test]
    fn test_state_differs_for_different_seeds() {
        let sphur1 = Sphur::new_seeded(1u64);
        let sphur2 = Sphur::new_seeded(2u64);

        assert_ne!(
            sphur1.state, sphur2.state,
            "State should differ for different seeds"
        );
    }

    #[test]
    fn test_no_zero_elements_after_seed() {
        let sphur = Sphur::new_seeded(987654321u64);

        for (i, &val) in sphur.state.0.iter().enumerate() {
            assert_ne!(val, 0, "State element {} should not be zero", i);
        }
    }

    #[test]
    fn test_init_state_matches_new_seeded() {
        let seed = 2025u64;
        let state_from_init = Sphur::init_state(seed);
        let sphur = Sphur::new_seeded(seed);

        assert_eq!(
            state_from_init, sphur.state,
            "init_state should match new_seeded output"
        );
    }
}

#[cfg(test)]
mod rand_gen_tests {
    use super::*;

    const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

    #[test]
    fn test_deterministic_u64_u128() {
        let mut rng1 = Sphur::new_seeded(TEST_SEED);
        let mut rng2 = Sphur::new_seeded(TEST_SEED);

        for _ in 0..100 {
            assert_eq!(
                rng1.gen_u64(),
                rng2.gen_u64(),
                "gen_u64 should be deterministic for same seed"
            );
            assert_eq!(
                rng1.gen_u128(),
                rng2.gen_u128(),
                "gen_u128 should be deterministic for same seed"
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

    #[test]
    fn test_full_u128_entropy() {
        let mut rng = Sphur::new_seeded(TEST_SEED);

        let mut words = [0u128; N_STATE];
        for i in 0..N_STATE {
            words[i] = rng.gen_u128();
        }

        // Ensure no duplicates in one state block
        for i in 0..N_STATE {
            for j in i + 1..N_STATE {
                assert_ne!(
                    words[i], words[j],
                    "all u128s in a block should be distinct"
                );
            }
        }
    }
}
