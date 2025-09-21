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
    pub fn new_seeded(seed: u64) -> Self {
        Self {
            state: Self::init_state(seed),
            isa: detect_isa(),
            idx: 0,
        }
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
}

/// Custom result type for [Sphur]
pub type SphurResult<T> = Result<T, SphurError>;

/// Types of Error exposed by [Sphur]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SphurError {
    InitError,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_isa_returns_valid_value() {
        match detect_isa() {
            ISA::AVX2 | ISA::SSE2 | ISA::NEON => {}
            _ => panic!("Unknown ISA detected"),
        }
    }

    #[test]
    fn test_detect_isa_is_correct() {
        let isa = detect_isa();

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
