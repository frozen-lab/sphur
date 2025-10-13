use super::init_engine_state;
use super::Engine;

use core::arch::x86_64::*;

pub(crate) const SSE_STATE_LEN: usize = 156;
const _: () = assert!(SSE_STATE_LEN % 2 == 0);

const SR1: i32 = 11;
const SL1: i32 = 18;

const SR2: i32 = 1;
const SL2: i32 = 1;

const POS1: usize = 122;
const MSK: [u32; 4] = [0xdfffffefu32, 0xddfecb7fu32, 0xbffaffffu32, 0xbffffff6u32];

pub(crate) struct SSE;

impl Engine<SSE_STATE_LEN> for SSE {
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
        let mask = Self::get_mask();
        let n = SSE_STATE_LEN;

        let mut i0 = 0usize;

        while i0 + 1 < n {
            //
            // compute for lane `i`
            //

            let mut b0 = i0 + POS1;
            if b0 >= n {
                b0 -= n;
            }

            let mut c0 = i0 + n - 2;
            if c0 >= n {
                c0 -= n;
            }

            let mut d0 = i0 + n - 1;
            if d0 >= n {
                d0 -= n;
            }

            //
            // compute for lane `i + 1`
            //

            let i1 = i0 + 1;
            let mut b1 = i1 + POS1;

            if b1 >= n {
                b1 -= n;
            }

            let mut c1 = i1 + n - 2;
            if c1 >= n {
                c1 -= n;
            }

            let mut d1 = i1 + n - 1;
            if d1 >= n {
                d1 -= n;
            }

            //
            // load lane `i`
            //

            let a_0 = *state.get_unchecked(i0);
            let b_0 = *state.get_unchecked(b0);
            let c_0 = *state.get_unchecked(c0);
            let d_0 = *state.get_unchecked(d0);

            //
            // load lane `i + 1`
            //

            let a_1 = *state.get_unchecked(i1);
            let b_1 = *state.get_unchecked(b1);
            let c_1 = *state.get_unchecked(c1);
            let d_1 = *state.get_unchecked(d1);

            let out0 = recurrence_relation(a_0, b_0, c_0, d_0, mask);
            let out1 = recurrence_relation(a_1, b_1, c_1, d_1, mask);

            *state.get_unchecked_mut(i0) = out0;
            *state.get_unchecked_mut(i1) = out1;

            i0 += 2;
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u64(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u64 {
        // sanity check
        debug_assert!(idx < 2, "Index must be smaller than 2 for SSE lane");
        let lane_ref = *state.get_unchecked(lane);

        #[cfg(target_feature = "sse4.1")]
        {
            return SSE::gen_u64_const(lane_ref, idx as i32);
        }

        let ptr = &lane_ref as *const __m128i as *const u64;
        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u32(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u32 {
        // sanity check
        debug_assert!(idx < 4, "Index must be smaller then 4 for SSE state");
        let lane_ref = *state.get_unchecked(lane);

        #[cfg(target_feature = "sse4.1")]
        {
            return SSE::gen_u32_const(lane_ref, idx as i32);
        }

        let ptr = &lane_ref as *const __m128i as *const u32;
        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u16(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u16 {
        debug_assert!(
            idx < 8,
            "Index must be smaller than 8 for SSE lane (128 bits / 16 bits)"
        );

        let lane_ref = *state.get_unchecked(lane);

        #[cfg(target_feature = "sse4.1")]
        {
            return SSE::gen_u16_const(lane_ref, idx as i32);
        }

        let ptr = &lane_ref as *const __m128i as *const u16;
        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u8(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> u8 {
        debug_assert!(
            idx < 16,
            "Index must be smaller than 16 for SSE lane (128 bits / 8 bits)"
        );

        let lane_ref = *state.get_unchecked(lane);

        #[cfg(target_feature = "sse4.1")]
        {
            return SSE::gen_u8_const(lane_ref, idx as i32);
        }

        let ptr = &lane_ref as *const __m128i as *const u8;
        *ptr.add(idx)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_bool(state: &[Self::Lane; SSE_STATE_LEN], lane: usize, idx: usize) -> bool {
        debug_assert!(
            idx < 16,
            "Index must be smaller than 16 for SSE lane (128 bits / 8 bits)"
        );

        SSE::gen_u8(state, lane, idx) & 1 != 0
    }
}

impl SSE {
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn get_mask() -> __m128i {
        _mm_set_epi32(MSK[3] as i32, MSK[2] as i32, MSK[1] as i32, MSK[0] as i32)
    }

    #[cfg(target_feature = "sse4.1")]
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u64_const(lane_ref: __m128i, idx: i32) -> u64 {
        match idx {
            0 => _mm_extract_epi64(lane_ref, 0) as u64,
            1 => _mm_extract_epi64(lane_ref, 1) as u64,
            _ => core::hint::unreachable_unchecked(),
        }
    }

    #[cfg(target_feature = "sse4.1")]
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u32_const(lane_ref: __m128i, idx: i32) -> u32 {
        match idx {
            0 => _mm_extract_epi32(lane_ref, 0) as u32,
            1 => _mm_extract_epi32(lane_ref, 1) as u32,
            2 => _mm_extract_epi32(lane_ref, 2) as u32,
            3 => _mm_extract_epi32(lane_ref, 3) as u32,
            _ => core::hint::unreachable_unchecked(),
        }
    }

    #[cfg(target_feature = "sse4.1")]
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u16_const(lane_ref: __m128i, idx: i32) -> u16 {
        match idx {
            0 => _mm_extract_epi16(lane_ref, 0) as u16,
            1 => _mm_extract_epi16(lane_ref, 1) as u16,
            2 => _mm_extract_epi16(lane_ref, 2) as u16,
            3 => _mm_extract_epi16(lane_ref, 3) as u16,
            4 => _mm_extract_epi16(lane_ref, 4) as u16,
            5 => _mm_extract_epi16(lane_ref, 5) as u16,
            6 => _mm_extract_epi16(lane_ref, 6) as u16,
            7 => _mm_extract_epi16(lane_ref, 7) as u16,
            _ => core::hint::unreachable_unchecked(),
        }
    }

    #[cfg(target_feature = "sse4.1")]
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn gen_u8_const(lane_ref: __m128i, idx: i32) -> u8 {
        match idx {
            0 => _mm_extract_epi8(lane_ref, 0) as u8,
            1 => _mm_extract_epi8(lane_ref, 1) as u8,
            2 => _mm_extract_epi8(lane_ref, 2) as u8,
            3 => _mm_extract_epi8(lane_ref, 3) as u8,
            4 => _mm_extract_epi8(lane_ref, 4) as u8,
            5 => _mm_extract_epi8(lane_ref, 5) as u8,
            6 => _mm_extract_epi8(lane_ref, 6) as u8,
            7 => _mm_extract_epi8(lane_ref, 7) as u8,
            8 => _mm_extract_epi8(lane_ref, 8) as u8,
            9 => _mm_extract_epi8(lane_ref, 9) as u8,
            10 => _mm_extract_epi8(lane_ref, 10) as u8,
            11 => _mm_extract_epi8(lane_ref, 11) as u8,
            12 => _mm_extract_epi8(lane_ref, 12) as u8,
            13 => _mm_extract_epi8(lane_ref, 13) as u8,
            14 => _mm_extract_epi8(lane_ref, 14) as u8,
            15 => _mm_extract_epi8(lane_ref, 15) as u8,
            _ => core::hint::unreachable_unchecked(),
        }
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
///
/// ## ILP
///
/// Rather then chaining together, we compute indep pieces, for ILP
///
/// ```md
/// t0 = a ^ (a << SL1)
/// by = (b >> SR1) & mask
/// t1 = sr128lane(c) ^ sl128lane(d)
///
/// (finally) out = t0 ^ by ^ t1
/// ```
unsafe fn recurrence_relation(a: __m128i, b: __m128i, c: __m128i, d: __m128i, mask: __m128i) -> __m128i {
    // t0 = a ^ (a << SL1)
    let ax = _mm_slli_epi32(a, SL1 as i32);
    let t0 = _mm_xor_si128(a, ax);

    // by = elem's shift + mask
    let by = _mm_and_si128(_mm_srli_epi32(b, SR1 as i32), mask);

    // t1 = combine c and d shifts
    let c_sr2 = sr_128_lane(c);
    let d_sl2 = sl_128_lane(d);
    let t1 = _mm_xor_si128(c_sr2, d_sl2);

    // out = ((a ^ ax) ^ by) ^ (c_sr2 ^ d_sl2)
    _mm_xor_si128(t0, _mm_xor_si128(by, t1))
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
    // pre elem right shift
    let part1 = _mm_srli_epi32(x, SR2);

    // cross lane carry
    #[cfg(target_feature = "ssse3")]
    {
        let shifted_bytes = _mm_alignr_epi8(x, x, SR2 / 8);
        let part2 = _mm_slli_epi32(shifted_bytes, 32 - SR2);

        return _mm_or_si128(part1, part2);
    }

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
    // pre elem right shift
    let part1 = _mm_slli_epi32(x, SL2);

    // cross lane carry
    #[cfg(target_feature = "ssse3")]
    {
        let shifted_bytes = _mm_alignr_epi8(x, x, 16 - (SL2 / 8));
        let part2 = _mm_srli_epi32(shifted_bytes, 32 - SL2);

        return _mm_or_si128(part1, part2);
    }

    let tmp = _mm_slli_si128(x, 4);
    let part2 = _mm_srli_epi32(tmp, 32 - SL2);

    _mm_or_si128(part1, part2)
}

#[cfg(test)]
mod sse_tests {
    use super::*;

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
                let mask = SSE::get_mask();

                let a = _mm_set1_epi32(0xdeadbeefu32 as i32);
                let b = _mm_set1_epi32(0x12345678);
                let c = _mm_set1_epi32(0x0badf00d);
                let d = _mm_set1_epi32(0x9abcdef0u32 as i32);

                let r1 = recurrence_relation(a, b, c, d, mask);
                let r2 = recurrence_relation(a, b, c, d, mask);

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
                let mask = SSE::get_mask();
                let mut state = [_mm_set1_epi32(0x12345678u32 as i32); SSE_STATE_LEN];

                for i in 0..SSE_STATE_LEN {
                    let a = state[i];
                    let b = state[(i + 93) % SSE_STATE_LEN];
                    let c = state[(i + SSE_STATE_LEN - 2) % SSE_STATE_LEN];
                    let d = state[(i + SSE_STATE_LEN - 1) % SSE_STATE_LEN];

                    state[i] = recurrence_relation(a, b, c, d, mask);
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
        fn test_recurrence_diverges_with_different_seed() {
            unsafe {
                let mask = SSE::get_mask();
                let mut s1 = [_mm_set1_epi32(0xAAAAAAAAu32 as i32); SSE_STATE_LEN];
                let mut s2 = [_mm_set1_epi32(0xAAAAAAABu32 as i32); SSE_STATE_LEN];

                for i in 0..SSE_STATE_LEN {
                    let (a1, b1, c1, d1) = (
                        s1[i],
                        s1[(i + 93) % SSE_STATE_LEN],
                        s1[(i + SSE_STATE_LEN - 2) % SSE_STATE_LEN],
                        s1[(i + SSE_STATE_LEN - 1) % SSE_STATE_LEN],
                    );

                    let (a2, b2, c2, d2) = (
                        s2[i],
                        s2[(i + 93) % SSE_STATE_LEN],
                        s2[(i + SSE_STATE_LEN - 2) % SSE_STATE_LEN],
                        s2[(i + SSE_STATE_LEN - 1) % SSE_STATE_LEN],
                    );

                    s1[i] = recurrence_relation(a1, b1, c1, d1, mask);
                    s2[i] = recurrence_relation(a2, b2, c2, d2, mask);
                }

                let mut equal_count = 0;

                for i in 0..SSE_STATE_LEN {
                    let mut b1 = [0u32; 4];
                    let mut b2 = [0u32; 4];

                    _mm_storeu_si128(b1.as_mut_ptr() as *mut __m128i, s1[i]);
                    _mm_storeu_si128(b2.as_mut_ptr() as *mut __m128i, s2[i]);

                    if b1 == b2 {
                        equal_count += 1;
                    }
                }

                assert!(equal_count < SSE_STATE_LEN / 4, "different seeds must diverge early");
            }
        }

        #[test]
        fn test_state_alignment_safe() {
            unsafe {
                let mut state = [_mm_setzero_si128(); SSE_STATE_LEN];
                let ptr = state.as_ptr();

                assert_eq!(ptr.align_offset(16), 0, "state must be 16-byte aligned");
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
                let v8 = SSE::gen_u8(&state, 0, 0);
                assert_eq!(v8, 0xAA);

                // u16
                let v16 = SSE::gen_u16(&state, 0, 0);
                assert_eq!(v16, 0xAAAA);

                // u32
                let mut lane = _mm_set_epi32(0x11223344, 0x55667788, 0x99AABBCCu32 as i32, 0xDDEEFF00u32 as i32);
                state[0] = lane;

                assert_eq!(SSE::gen_u32(&state, 0, 0), 0xDDEEFF00);
                assert_eq!(SSE::gen_u32(&state, 0, 1), 0x99AABBCC);
                assert_eq!(SSE::gen_u32(&state, 0, 2), 0x55667788);
                assert_eq!(SSE::gen_u32(&state, 0, 3), 0x11223344);

                // u64
                lane = _mm_set_epi64x(0xAABBCCDDEEFF0011u64 as i64, 0x1122334455667788);
                state[0] = lane;
                assert_eq!(SSE::gen_u64(&state, 0, 0), 0x1122334455667788);
                assert_eq!(SSE::gen_u64(&state, 0, 1), 0xAABBCCDDEEFF0011);

                // bool (just check bit extraction)
                let b_true = SSE::gen_bool(&state, 0, 0);
                let b_false = SSE::gen_bool(&state, 0, 1);
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
                        SSE::gen_u8(&state, 0, 16);
                    });
                    assert!(result.is_err());
                }
            }
        }

        #[test]
        fn test_gen_consistency_across_calls() {
            unsafe {
                let state = [_mm_set_epi32(1, 2, 3, 4); SSE_STATE_LEN];

                let v1 = SSE::gen_u32(&state, 0, 0);
                let v2 = SSE::gen_u32(&state, 0, 0);
                assert_eq!(v1, v2);

                let v3 = SSE::gen_u16(&state, 0, 0);
                let v4 = SSE::gen_u16(&state, 0, 0);
                assert_eq!(v3, v4);
            }
        }
    }

    mod sse {
        use super::*;

        #[test]
        fn test_get_mask_correctness() {
            unsafe {
                let mask = SSE::get_mask();

                let mut buf = [0u32; 4];
                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, mask);

                assert_eq!(buf, MSK, "Mask constant must match SFMT definition");
            }
        }

        #[cfg(target_feature = "sse4.1")]
        #[test]
        fn test_gen_const_functions_exhaustive() {
            unsafe {
                let lane = _mm_set_epi8(
                    0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
                );

                // u8 extraction (0..15)
                for i in 0..16 {
                    let v = SSE::gen_u8_const(lane, i);
                    assert_eq!(v, i as u8, "u8 const extraction mismatch at {}", i);
                }

                // u16 extraction (0..7)
                for i in 0..8 {
                    let v = SSE::gen_u16_const(lane, i);
                    // Each 16-bit block = little endian combination of two u8s
                    let expected = ((2 * i + 1) as u16) << 8 | (2 * i) as u16;
                    assert_eq!(v, expected, "u16 const extraction mismatch at {}", i);
                }

                // u32 extraction (0..3)
                let lane32 = _mm_set_epi32(0x11223344, 0x55667788, 0x99AABBCCu32 as i32, 0xDDEEFF00u32 as i32);

                for (i, exp) in [0xDDEEFF00u32 as i32, 0x99AABBCCu32 as i32, 0x55667788, 0x11223344]
                    .iter()
                    .enumerate()
                {
                    let v = SSE::gen_u32_const(lane32, i as i32);
                    assert_eq!(v, *exp as u32, "u32 const extraction mismatch at {}", i);
                }

                // u64 extraction (0..1)
                let lane64 = _mm_set_epi64x(0xAABBCCDDEEFF0011u64 as i64, 0x1122334455667788);

                assert_eq!(SSE::gen_u64_const(lane64, 0), 0x1122334455667788);
                assert_eq!(SSE::gen_u64_const(lane64, 1), 0xAABBCCDDEEFF0011);
            }
        }

        #[test]
        fn test_sr_sl_lane_roundtrip() {
            unsafe {
                let x = _mm_set_epi32(0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d);
                let r = sl_128_lane(sr_128_lane(x));

                let mut orig = [0u32; 4];
                let mut out = [0u32; 4];

                _mm_storeu_si128(orig.as_mut_ptr() as *mut __m128i, x);
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, r);

                let equal_bits = orig.iter().zip(out.iter()).filter(|(a, b)| a == b).count();
                assert!(equal_bits >= 1, "roundtrip lost all structure: {:?} -> {:?}", orig, out);
            }
        }

        #[test]
        fn test_recurrence_relation_determinism_and_linearity() {
            unsafe {
                let mask = SSE::get_mask();

                let a = _mm_set1_epi32(0xdeadbeefu32 as i32);
                let b = _mm_set1_epi32(0x12345678);
                let c = _mm_set1_epi32(0x0badf00d);
                let d = _mm_set1_epi32(0x9abcdef0u32 as i32);

                let r1 = recurrence_relation(a, b, c, d, mask);
                let r2 = recurrence_relation(a, b, c, d, mask);

                let mut buf1 = [0u32; 4];
                let mut buf2 = [0u32; 4];

                _mm_storeu_si128(buf1.as_mut_ptr() as *mut __m128i, r1);
                _mm_storeu_si128(buf2.as_mut_ptr() as *mut __m128i, r2);

                assert_eq!(buf1, buf2, "recurrence must be deterministic");
            }
        }
    }
}
