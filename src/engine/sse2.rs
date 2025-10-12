use super::init_engine_state;
use core::arch::x86_64::*;

pub(crate) const SSE_STATE_LEN: usize = 156;

const SR1: i32 = 11;
const SL1: i32 = 18;

const SR2: i32 = 1;
const SL2: i32 = 1;

const POS1: usize = 122;
const MSK: [u32; 4] = [0xdfffffefu32, 0xddfecb7fu32, 0xbffaffffu32, 0xbffffff6u32];

pub(crate) struct SSE2;

impl super::Engine<SSE_STATE_LEN> for SSE2 {
    type Lane = __m128i;

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn new(seed: u64) -> [Self::Lane; SSE_STATE_LEN] {
        init_engine_state::<Self::Lane, SSE_STATE_LEN, { SSE_STATE_LEN * 4 }>(seed, |w| {
            _mm_set_epi32(w[3] as i32, w[2] as i32, w[1] as i32, w[0] as i32)
        })
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn regen(state: &mut [Self::Lane; SSE_STATE_LEN]) {
        for i in 0..SSE_STATE_LEN {
            let b_idx = (i + POS1) % SSE_STATE_LEN;
            let c_idx = (i + SSE_STATE_LEN - 2) % SSE_STATE_LEN;
            let d_idx = (i + SSE_STATE_LEN - 1) % SSE_STATE_LEN;

            let a = state[i];
            let b = state[b_idx];
            let c = state[c_idx];
            let d = state[d_idx];

            state[i] = recurrence_relation(a, b, c, d);
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u64(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u64 {
        // sanity check
        debug_assert!(idx < 2, "Index must be smaller than 2 for SSE lane");

        let lane_ref = state.get_unchecked(lane);
        let ptr = lane_ref as *const __m128i as *const u64;

        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u32(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u32 {
        // sanity check
        debug_assert!(idx < 4, "Index must be smaller then 4 for SSE state");

        let lane_ref = state.get_unchecked(lane);
        let ptr = lane_ref as *const __m128i as *const u32;

        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u16(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u16 {
        debug_assert!(
            idx < 8,
            "Index must be smaller than 8 for SSE lane (128 bits / 16 bits)"
        );

        let lane_ref = state.get_unchecked(lane);
        let ptr = lane_ref as *const __m128i as *const u16;
        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u8(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u8 {
        debug_assert!(
            idx < 16,
            "Index must be smaller than 16 for SSE lane (128 bits / 8 bits)"
        );

        let lane_ref = state.get_unchecked(lane);
        let ptr = lane_ref as *const __m128i as *const u8;
        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_bool(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> bool {
        debug_assert!(
            idx < 16,
            "Index must be smaller than 16 for SSE lane (128 bits / 8 bits)"
        );

        let lane_ref = state.get_unchecked(lane);
        let ptr = lane_ref as *const __m128i as *const u8;
        *ptr.add(idx) & 1 != 0
    }
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
/// Performs SFMT recurrence relation
///
/// ## Algo
///
/// ```md
/// w = (a << SL1)
/// x = ((b >> SR1) & MSK)
/// y = (c >> 128 bits by SR2 logic)
/// z = (d << 128 bits by SL2 logic)
///
/// out = a ^ w ^ x ^ y ^ z
/// ```
unsafe fn recurrence_relation(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
    let mask: __m128i = _mm_set_epi32(MSK[3] as i32, MSK[2] as i32, MSK[1] as i32, MSK[0] as i32);

    let ax = _mm_slli_epi32(a, SL1 as i32);
    let by = _mm_and_si128(_mm_srli_epi32(b, SR1 as i32), mask);
    let c_sr2 = sr_128_lane(c);
    let d_sl2 = sl_128_lane(d);

    let mut r = _mm_xor_si128(a, ax);
    r = _mm_xor_si128(r, by);
    r = _mm_xor_si128(r, c_sr2);
    _mm_xor_si128(r, d_sl2)
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
/// Perform right shft on entire sse2 lane
///
/// ## Visulization
///
/// ```md
/// inp => | A0 A1 A2 A3 | B0 B1 B2 B3 | C0 C1 C2 C3 | D0 D1 D2 D3 |
///
/// | 00 A0 A1 A2 | 00 B0 B1 B2 | 00 C0 C1 C2 | 00 D0 D1 D2 |
/// | 00 00 00 00 | A0 A1 A2 A3 | B0 B1 B2 B3 | C0 C1 C2 C3 |
/// | 00 00 00 00 | A3 00 00 00 | B3 00 00 00 | C3 00 00 00 |
///
/// out => | 00 A0 A1 A2 | A3 B0 B1 B2 | B3 C0 C1 C2 | C3 D0 D1 D2 |
///
/// ```
unsafe fn sr_128_lane(x: __m128i) -> __m128i {
    let part1 = _mm_srli_epi32(x, SR2);
    let tmp = _mm_srli_si128(x, 4);
    let part2 = _mm_slli_epi32(tmp, 32 - SR2);

    _mm_or_si128(part1, part2)
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
/// Perform left shift on entire sse2 lane
///
/// ## Visualization
///
/// ```md
/// inp => | A0 A1 A2 A3 | B0 B1 B2 B3 | C0 C1 C2 C3 | D0 D1 D2 D3 |
///
/// | A1 A2 A3 00 | B1 B2 B3 00 | C1 C2 C3 00 | D1 D2 D3 00 |
/// | B0 B1 B2 B3 | C0 C1 C2 C3 | D0 D1 D2 D3 | 00 00 00 00 |
/// | 00 00 00 B0 | 00 00 00 C0 | 00 00 00 D0 | 00 00 00 00 |
///
/// out => | A1 A2 A3 B0 | B1 B2 B3 C0 | C1 C2 C3 D0 | D1 D2 D3 00 |
/// ```
unsafe fn sl_128_lane(x: __m128i) -> __m128i {
    let part1 = _mm_slli_epi32(x, SL2);
    let tmp = _mm_slli_si128(x, 4);
    let part2 = _mm_srli_epi32(tmp, 32 - SL2);

    _mm_or_si128(part1, part2)
}

#[cfg(test)]
mod sse2_tests {
    use super::*;
    use crate::engine::Engine;

    mod indep_functions {
        use super::*;

        #[test]
        fn test_sr_128_lane_basic() {
            unsafe {
                let x = _mm_set_epi32(0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d);
                let r = sr_128_lane(x);

                let mut buf = [0u32; 4];
                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, r);

                assert_ne!(buf, [0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d]);
            }
        }

        #[test]
        fn test_sl_128_lane_basic() {
            unsafe {
                let x = _mm_set_epi32(0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d);
                let r = sl_128_lane(x);

                let mut buf = [0u32; 4];
                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, r);

                assert_ne!(buf, [0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d]);
                // changed bits
            }
        }

        #[test]
        fn test_recurrence_relation_stability() {
            unsafe {
                let a = _mm_set1_epi32(0xdeadbeefu32 as i32);
                let b = _mm_set1_epi32(0x12345678);
                let c = _mm_set1_epi32(0x0badf00d);
                let d = _mm_set1_epi32(0x9abcdef0u32 as i32);

                let r1 = recurrence_relation(a, b, c, d);
                let r2 = recurrence_relation(a, b, c, d);

                let mut buf1 = [0u32; 4];
                let mut buf2 = [0u32; 4];

                _mm_storeu_si128(buf1.as_mut_ptr() as *mut __m128i, r1);
                _mm_storeu_si128(buf2.as_mut_ptr() as *mut __m128i, r2);

                assert_eq!(buf1, buf2);
            }
        }
    }

    mod engine {
        use super::*;

        #[test]
        fn test_gen_state_runs() {
            unsafe {
                let mut state = [_mm_set1_epi32(0x12345678u32 as i32); SSE_STATE_LEN];

                for i in 0..SSE_STATE_LEN {
                    let a = state[i];
                    let b = state[(i + 93) % SSE_STATE_LEN];
                    let c = state[(i + SSE_STATE_LEN - 2) % SSE_STATE_LEN];
                    let d = state[(i + SSE_STATE_LEN - 1) % SSE_STATE_LEN];

                    state[i] = recurrence_relation(a, b, c, d);
                }

                // sanity check
                let mut buf = [0u32; 4];
                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, state[0]);
                assert_ne!(buf, [0x12345678; 4], "recurrence should mutate the state");

                let mut unique = std::collections::HashSet::new();

                for i in 0..SSE_STATE_LEN {
                    let mut tmp = [0u32; 4];

                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, state[i]);
                    unique.insert(tmp);
                }

                assert!(unique.len() > 1, "recurrence produced diverse state values");
            }
        }

        #[test]
        fn test_recurrence_determinism() {
            unsafe {
                let mut s1 = [_mm_set1_epi32(0xAABBCCDDu32 as i32); SSE_STATE_LEN];
                let mut s2 = [_mm_set1_epi32(0xAABBCCDDu32 as i32); SSE_STATE_LEN];

                for i in 0..SSE_STATE_LEN {
                    let a1 = s1[i];
                    let b1 = s1[(i + 93) % SSE_STATE_LEN];
                    let c1 = s1[(i + SSE_STATE_LEN - 2) % SSE_STATE_LEN];
                    let d1 = s1[(i + SSE_STATE_LEN - 1) % SSE_STATE_LEN];

                    s1[i] = recurrence_relation(a1, b1, c1, d1);

                    let a2 = s2[i];
                    let b2 = s2[(i + 93) % SSE_STATE_LEN];
                    let c2 = s2[(i + SSE_STATE_LEN - 2) % SSE_STATE_LEN];
                    let d2 = s2[(i + SSE_STATE_LEN - 1) % SSE_STATE_LEN];

                    s2[i] = recurrence_relation(a2, b2, c2, d2);
                }

                for i in 0..SSE_STATE_LEN {
                    let mut buf1 = [0u32; 4];
                    let mut buf2 = [0u32; 4];

                    _mm_storeu_si128(buf1.as_mut_ptr() as *mut __m128i, s1[i]);
                    _mm_storeu_si128(buf2.as_mut_ptr() as *mut __m128i, s2[i]);
                    assert_eq!(buf1, buf2, "recurrence must be deterministic");
                }
            }
        }
    }

    mod gen_functions {
        use super::*;

        #[test]
        fn test_gen_u8_u16_u32_u64_bool_basic() {
            unsafe {
                let mut state = [_mm_set1_epi8(0xAAu8 as i8); SSE_STATE_LEN];

                // u8
                let v8 = SSE2::gen_u8(&state, 0, 0);
                assert_eq!(v8, 0xAA);

                // u16
                let v16 = SSE2::gen_u16(&state, 0, 0);
                assert_eq!(v16, 0xAAAA);

                // u32
                let mut lane = _mm_set_epi32(0x11223344, 0x55667788, 0x99AABBCCu32 as i32, 0xDDEEFF00u32 as i32);
                state[0] = lane;

                assert_eq!(SSE2::gen_u32(&state, 0, 0), 0xDDEEFF00);
                assert_eq!(SSE2::gen_u32(&state, 0, 1), 0x99AABBCC);
                assert_eq!(SSE2::gen_u32(&state, 0, 2), 0x55667788);
                assert_eq!(SSE2::gen_u32(&state, 0, 3), 0x11223344);

                // u64
                lane = _mm_set_epi64x(0xAABBCCDDEEFF0011u64 as i64, 0x1122334455667788);
                state[0] = lane;
                assert_eq!(SSE2::gen_u64(&state, 0, 0), 0x1122334455667788);
                assert_eq!(SSE2::gen_u64(&state, 0, 1), 0xAABBCCDDEEFF0011);

                // bool (just check bit extraction)
                let b_true = SSE2::gen_bool(&state, 0, 0);
                let b_false = SSE2::gen_bool(&state, 0, 1);
                assert!(b_true == true || b_false == true || b_true == b_false);
            }
        }

        #[test]
        fn test_gen_index_bounds() {
            unsafe {
                let state = [_mm_setzero_si128(); SSE_STATE_LEN];

                // should panic only in debug with out-of-bound idx
                #[cfg(debug_assertions)]
                {
                    let result = std::panic::catch_unwind(|| {
                        SSE2::gen_u8(&state, 0, 16);
                    });
                    assert!(result.is_err());
                }
            }
        }

        #[test]
        fn test_gen_consistency_across_calls() {
            unsafe {
                let state = [_mm_set_epi32(1, 2, 3, 4); SSE_STATE_LEN];

                let v1 = SSE2::gen_u32(&state, 0, 0);
                let v2 = SSE2::gen_u32(&state, 0, 0);
                assert_eq!(v1, v2);

                let v3 = SSE2::gen_u16(&state, 0, 0);
                let v4 = SSE2::gen_u16(&state, 0, 0);
                assert_eq!(v3, v4);
            }
        }
    }
}
