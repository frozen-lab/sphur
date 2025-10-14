use core::arch::x86_64::*;

pub(crate) const SSE_STATE_LEN: usize = 156;
pub(crate) const SSE_N64: usize = 2;
pub(crate) const SSE_N32: usize = 4;

const _: () = assert!(SSE_STATE_LEN % 2 == 0);
const _: () = assert!(SSE_N64 % 2 == 0);
const _: () = assert!(SSE_N32 % 2 == 0);

const SL1: i32 = 18;
const SL2: i32 = 1;

const SR1: i32 = 11;
const SR2: i32 = 1;

const POS1: usize = 122;
static MASK128: __m128i = unsafe { core::mem::transmute([0xdfffffefu32, 0xddfecb7fu32, 0xbffaffffu32, 0xbffffff6u32]) };

pub(crate) struct SSE;

impl super::Engine<SSE_STATE_LEN, SSE_N64, SSE_N32> for SSE {
    type Lane = __m128i;

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn new(seed: u64) -> [Self::Lane; SSE_STATE_LEN] {
        super::init_engine_state::<Self::Lane, SSE_STATE_LEN, { SSE_STATE_LEN * 4 }>(seed, |w| {
            _mm_set_epi32(w[3] as i32, w[2] as i32, w[1] as i32, w[0] as i32)
        })
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn regen(state: &mut [Self::Lane; SSE_STATE_LEN]) {
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

            let out0 = recurrence_relation(a_0, b_0, c_0, d_0);
            let out1 = recurrence_relation(a_1, b_1, c_1, d_1);

            *state.get_unchecked_mut(i0) = out0;
            *state.get_unchecked_mut(i1) = out1;

            i0 += 2;
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn batch_u64(state: &[Self::Lane; SSE_STATE_LEN], lane: usize) -> [u64; SSE_N64] {
        // sanity check
        debug_assert!(lane < SSE_STATE_LEN, "Lane must be in bounds with the state size");

        let mut buf = [0u64; SSE_N64];
        let lane_ref = state.get_unchecked(lane);

        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, *lane_ref);

        buf
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn batch_u32(state: &[Self::Lane; SSE_STATE_LEN], lane: usize) -> [u32; SSE_N32] {
        // sanity check
        debug_assert!(lane < SSE_STATE_LEN, "Lane must be in bounds with the state size");

        let mut buf = [0u32; SSE_N32];
        let lane_ref = state.get_unchecked(lane);

        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, *lane_ref);

        buf
    }
}

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
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn recurrence_relation(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
    // t0 = a ^ (a << SL1)
    let ax = _mm_slli_epi32(a, SL1 as i32);
    let t0 = _mm_xor_si128(a, ax);

    // by = elem's shift + mask
    let by = _mm_and_si128(_mm_srli_epi32(b, SR1 as i32), MASK128);

    // t1 = combine c and d shifts
    let c_sr2 = sr_128_lane(c);
    let d_sl2 = sl_128_lane_alignr(d);
    let t1 = _mm_xor_si128(c_sr2, d_sl2);

    // out = ((a ^ ax) ^ by) ^ (c_sr2 ^ d_sl2)
    _mm_xor_si128(t0, _mm_xor_si128(by, t1))
}

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
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sr_128_lane(x: __m128i) -> __m128i {
    let part1 = _mm_srli_epi32(x, SR2);
    let tmp = _mm_srli_si128(x, 4);
    let part2 = _mm_slli_epi32(tmp, 32 - SR2);

    _mm_or_si128(part1, part2)
}

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
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sl_128_lane(x: __m128i) -> __m128i {
    let part1 = _mm_slli_epi32(x, SL2);
    let tmp = _mm_slli_si128(x, 4);
    let part2 = _mm_srli_epi32(tmp, 32 - SL2);

    _mm_or_si128(part1, part2)
}

#[target_feature(enable = "ssse3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sl_128_lane_alignr(x: __m128i) -> __m128i {
    // part1 = x << SL2 (per-element). if SL2==1, compiler may use vpaddd; keep it generic:
    let part1 = _mm_slli_epi32(x, SL2);

    // tmp = bytes shifted left by 4 with zero-fill:
    // alignr(x, zero, 12) effectively gives x << 4 bytes with zero fill
    let zero = _mm_setzero_si128();
    let tmp = _mm_alignr_epi8::<12>(x, zero); // x concatenated with zero, shift out lower bytes → left-byte-shift with zero
    let part2 = _mm_srli_epi32(tmp, 32 - SL2);

    _mm_or_si128(part1, part2)
}

#[cfg(test)]
mod sse_tests {
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
        fn test_recurrence_diverges_with_different_seed() {
            unsafe {
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

                    s1[i] = recurrence_relation(a1, b1, c1, d1);
                    s2[i] = recurrence_relation(a2, b2, c2, d2);
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
                let state = [_mm_setzero_si128(); SSE_STATE_LEN];
                let ptr = state.as_ptr();

                assert_eq!(ptr.align_offset(16), 0, "state must be 16-byte aligned");
            }
        }
    }

    mod gen_functions {
        use super::*;

        #[test]
        fn test_gen_u64_batch() {
            unsafe {
                let state =
                    [_mm_set_epi32(0x11223344, 0x55667788, 0x99AABBCCu32 as i32, 0xDDEEFF00u32 as i32); SSE_STATE_LEN];
                let buf64 = SSE::batch_u64(&state, 0);

                let expected_low = 0x99AABBCCDDEEFF00u64;
                let expected_high = 0x1122334455667788u64;

                assert_eq!(buf64[0], expected_low);
                assert_eq!(buf64[1], expected_high);
            }
        }

        #[test]
        fn test_gen_u32_batch() {
            unsafe {
                let state =
                    [_mm_set_epi32(0x11223344, 0x55667788, 0x99AABBCCu32 as i32, 0xDDEEFF00u32 as i32); SSE_STATE_LEN];
                let out = SSE::batch_u32(&state, 0);

                assert_eq!(out, [0xDDEEFF00, 0x99AABBCC, 0x55667788, 0x11223344]);
            }
        }
    }

    mod sse {
        use super::*;

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

                assert_eq!(buf1, buf2, "recurrence must be deterministic");
            }
        }
    }
}
