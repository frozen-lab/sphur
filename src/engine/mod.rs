#[cfg(target_arch = "x86_64")]
pub(crate) mod sse;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

const PARITY: [u32; 4] = [0x00000001, 0x00000000, 0x00000000, 0x13c9e684];

pub(crate) trait Engine<const N: usize, const N64: usize, const N32: usize> {
    type Lane;

    unsafe fn new(seed: u64) -> [Self::Lane; N];
    unsafe fn regen(state: &mut [Self::Lane; N]);

    unsafe fn batch_u64(state: &[Self::Lane; N], lane: usize) -> [u64; N64];
    unsafe fn batch_u32(state: &[Self::Lane; N], lane: usize) -> [u32; N32];
}

#[inline(always)]
pub(super) fn init_engine_state<Lane, const N: usize, const TOTAL_WORDS: usize>(
    seed: u64,
    pack: impl Fn([u32; 4]) -> Lane,
) -> [Lane; N] {
    // NOTE: Sanity check cause the manual loop unroll assumes this condition
    debug_assert!(TOTAL_WORDS % 4 == 0, "TOTAL_WORDS must be multiple of 4");

    let mut s = [0u32; TOTAL_WORDS];

    s[0] = seed as u32;
    s[1] = (seed >> 32) as u32;

    const MULT: u32 = 1812433253u32;

    //
    // NOTE: Manual 4 way unroll for throughput
    //

    let mut i = 2usize;

    while i + 3 < TOTAL_WORDS {
        let prev0 = s[i - 1];
        let t0 = prev0 ^ (prev0 >> 30);
        s[i] = t0.wrapping_mul(MULT).wrapping_add(i as u32);

        let prev1 = s[i];
        let t1 = prev1 ^ (prev1 >> 30);
        s[i + 1] = t1.wrapping_mul(MULT).wrapping_add((i + 1) as u32);

        let prev2 = s[i + 1];
        let t2 = prev2 ^ (prev2 >> 30);
        s[i + 2] = t2.wrapping_mul(MULT).wrapping_add((i + 2) as u32);

        let prev3 = s[i + 2];
        let t3 = prev3 ^ (prev3 >> 30);
        s[i + 3] = t3.wrapping_mul(MULT).wrapping_add((i + 3) as u32);

        i += 4;
    }

    period_certify(&mut s);

    core::array::from_fn(|idx| {
        let base = idx * 4;
        pack([s[base], s[base + 1], s[base + 2], s[base + 3]])
    })
}

#[inline(always)]
fn period_certify<const TOTAL_WORDS: usize>(state: &mut [u32; TOTAL_WORDS]) {
    let inner = state[0] ^ state[1] ^ state[2] ^ state[3];
    let check = (inner & PARITY[0]) ^ (inner & PARITY[1]) ^ (inner & PARITY[2]) ^ (inner & PARITY[3]);

    if check.count_ones() & 1 == 0 {
        for i in 0..4 {
            let p = PARITY[i];

            if p != 0 {
                let bit = p.trailing_zeros();
                state[i] ^= 1u32 << bit;

                break;
            }
        }
    }
}

#[cfg(test)]
mod init_engine_state {
    use super::*;

    fn pack_identity(v: [u32; 4]) -> [u32; 4] {
        v
    }

    #[test]
    fn test_deterministic_for_same_seed() {
        const N: usize = 2;
        const TOTAL_WORDS: usize = 8;

        let a = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(123456789, pack_identity);
        let b = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(123456789, pack_identity);

        assert_eq!(a, b, "same seed must produce same output");
    }

    #[test]
    fn test_different_seed_produces_different_state() {
        const N: usize = 2;
        const TOTAL_WORDS: usize = 8;

        let a = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(1, pack_identity);
        let b = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(2, pack_identity);

        assert_ne!(a, b, "different seeds must produce different states");
    }

    #[test]
    fn test_output_shape_correct() {
        const N: usize = 4;
        const TOTAL_WORDS: usize = N * 4;

        let state = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(42, pack_identity);
        assert_eq!(state.len(), N);

        for lane in &state {
            assert_eq!(lane.len(), 4);
        }
    }

    #[test]
    fn test_seed_low_high_bits_stored_correctly() {
        const N: usize = 1;
        const TOTAL_WORDS: usize = 4;

        let seed = 0x1122_3344_5566_7788u64;
        let state = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(seed, pack_identity);

        let flat = state[0];

        assert_eq!(flat[0], 0x5566_7788u32 as u32, "low 32 bits should be stored first");
        assert_eq!(flat[1], 0x1122_3344u32 as u32, "high 32 bits should be stored second");
    }

    #[test]
    fn test_handles_small_and_large_seeds() {
        const N: usize = 1;
        const TOTAL_WORDS: usize = 4;

        let _zero = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(0, pack_identity);
        let _one = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(1, pack_identity);
        let _max = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(u64::MAX, pack_identity);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TOTAL_WORDS must be multiple of 4")]
    fn test_debug_assert_triggers_for_non_multiple_of_4() {
        const N: usize = 2;
        const TOTAL_WORDS: usize = 6;

        let _ = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(123, pack_identity);
    }

    #[test]
    fn test_smallest_valid_config() {
        const N: usize = 1;
        const TOTAL_WORDS: usize = 4;

        let state = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(123, pack_identity);

        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_large_config_runs_without_panic() {
        const N: usize = 156;
        const TOTAL_WORDS: usize = N * 4;

        let _ = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(0xABCDEF1234567890, pack_identity);
    }

    #[test]
    fn test_no_panic_on_large_random_seed() {
        const N: usize = 4;
        const TOTAL_WORDS: usize = N * 4;

        let _ = init_engine_state::<[u32; 4], N, TOTAL_WORDS>(u64::MAX - 12345, pack_identity);
    }
}
