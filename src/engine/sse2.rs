use super::init_engine_state;
use core::arch::x86_64::*;

pub(crate) const SSE2_STATE_LEN: usize = 156;

const SR1: i32 = 11;
const SL1: i32 = 18;

const SR2: i32 = 1;
const SL2: i32 = 1;

const POS1: usize = 122;
const MSK: [u32; 4] = [0xdfffffefu32, 0xddfecb7fu32, 0xbffaffffu32, 0xbffffff6u32];

pub(crate) struct SSE2;

impl super::Engine<SSE2_STATE_LEN> for SSE2 {
    type Lane = __m128i;

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn new(seed: u64) -> [Self::Lane; SSE2_STATE_LEN] {
        init_engine_state::<Self::Lane, SSE2_STATE_LEN, { SSE2_STATE_LEN * 4 }>(seed, |w| {
            _mm_set_epi32(w[3] as i32, w[2] as i32, w[1] as i32, w[0] as i32)
        })
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn regen(state: &mut [Self::Lane; SSE2_STATE_LEN]) {
        for i in 0..SSE2_STATE_LEN {
            let b_idx = (i + POS1) % SSE2_STATE_LEN;
            let c_idx = (i + SSE2_STATE_LEN - 2) % SSE2_STATE_LEN;
            let d_idx = (i + SSE2_STATE_LEN - 1) % SSE2_STATE_LEN;

            let a = state[i];
            let b = state[b_idx];
            let c = state[c_idx];
            let d = state[d_idx];

            state[i] = recurrence_relation(a, b, c, d);
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
}
