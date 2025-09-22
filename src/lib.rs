#![allow(unused)]

const N_STATE: usize = 16;

/// Sphūr is SIMD accelerated Pseudo-Random Number Generator.
#[derive(Debug, Clone, Copy)]
pub struct Sphur {
    /// Internal state for SFMT twister
    state: [u128; N_STATE],

    /// Isa (Instruction Set Architecture) available at runtime
    isa: ISA,

    /// Current idx of the state being used for twister
    idx: usize,
}

impl Sphur {
    pub fn new() -> Self {
        let seed = Self::platform_seed();

        Self {
            state: Self::init_state(seed),
            isa: ISA::detect_isa(),
            idx: 0,
        }
    }

    pub fn new_seeded(seed: u64) -> Self {
        Self {
            state: Self::init_state(seed),
            isa: ISA::detect_isa(),
            idx: 0,
        }
    }

    pub fn gen_u128(&mut self) -> u128 {
        if self.idx >= N_STATE {
            self.update_state();
        }

        let v = self.state[self.idx];
        self.idx += 1;

        v
    }

    /// Generates a batch of 32 u64 values from the current state.
    ///
    /// NOTE: This consumes the entire state (16 * 128-bit).
    pub fn gen_batch(&mut self) -> [u64; N_STATE * 2] {
        // Initial twist for a new batch
        self.update_state();

        let mut out = [0u64; N_STATE * 2];

        for (i, word) in self.state.iter().enumerate() {
            out[2 * i] = (*word & 0xffff_ffff_ffff_ffff) as u64;
            out[2 * i + 1] = (word >> 64) as u64;
        }

        out
    }

    /// Generate a random `u64` inside an exclusive range [start, end).
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

    pub fn gen_u64(&mut self) -> u64 {
        if self.idx >= N_STATE * 2 {
            self.update_state();
        }

        let word = self.state[self.idx >> 1]; // idx/2
        let hi = (word >> 64) as u64;
        let lo = word as u64;

        // mask = 0 if even, !0 if odd
        let mask = -((self.idx & 1) as i64) as u64;

        let out = (lo & !mask) | (hi & mask);

        self.idx += 1;
        out
    }

    pub fn gen_u32(&mut self) -> u32 {
        if self.idx >= N_STATE * 4 {
            self.update_state();
            self.idx = 0;
        }

        // idx / 4 -> u128
        let word = self.state[self.idx >> 2];

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
    #[inline(always)]
    pub fn gen_bool(&mut self) -> bool {
        // just grab 1 random bit
        (self.gen_u64() & 1) != 0
    }

    fn init_state(seed: u64) -> [u128; N_STATE] {
        let mut state = [0u128; 16];

        // initial seed
        state[0] = seed as u128;

        for i in 1..N_STATE {
            let prev = state[i - 1];

            state[i] = 6364136223846793005u128
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u128);
        }

        state
    }

    fn update_state(&mut self) {
        match self.isa {
            ISA::AVX2 => unsafe { sse2::twist_block(&mut self.state) },
            ISA::SSE2 => unsafe { sse2::twist_block(&mut self.state) },
            ISA::NEON => {}
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
    use super::N_STATE;
    use std::arch::x86_64::*;

    const POS1: usize = 122 % N_STATE;
    const SL1: i32 = 18;
    const SR1: i32 = 11;
    const SL2: i32 = 1;
    const SR2: i32 = 1;

    #[target_feature(enable = "sse2")]
    pub fn twist_block(state: &mut [u128; N_STATE]) {
        // sanity check
        debug_assert_eq!(std::mem::size_of::<u128>(), std::mem::size_of::<__m128i>());

        unsafe {
            let ptr = state.as_mut_ptr() as *mut __m128i;
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

/// Custom result type for [Sphur]
pub type SphurResult<T> = Result<T, SphurError>;

/// Types of Error exposed by [Sphur]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SphurError {}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
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
mod init_tests {
    use super::*;

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
            sphur.state.iter().any(|&x| x != 0),
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

        assert_eq!(sphur.state[0], seed as u128);
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

        for (i, &val) in sphur.state.iter().enumerate() {
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
