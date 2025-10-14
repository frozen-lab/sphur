use core::arch::x86_64::*;
use std::sync::Once;

pub(crate) const SSE_STATE_LEN: usize = 156;
pub(crate) const SSE_N64: usize = 2;
pub(crate) const SSE_N32: usize = 4;

const _: () = assert!(SSE_STATE_LEN % 2 == 0);
const _: () = assert!(SSE_N64 % 2 == 0);
const _: () = assert!(SSE_N32 % 2 == 0);

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
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn recurrence_relation(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
    // t0 = a ^ (a << SL1)
    let ax = _mm_slli_epi32(a, SL1 as i32);
    let t0 = _mm_xor_si128(a, ax);

    // by = elem's shift + mask
    let by = _mm_and_si128(_mm_srli_epi32(b, SR1 as i32), MASK128);

    // t1 = combine c and d shifts
    let c_sr2 = sr_128_lane_sse(c);
    let d_sl2 = sl_128_lane_sse(d);
    let t1 = _mm_xor_si128(c_sr2, d_sl2);

    // out = ((a ^ ax) ^ by) ^ (c_sr2 ^ d_sl2)
    _mm_xor_si128(t0, _mm_xor_si128(by, t1))
}

//
// shift right
//

const SR1: i32 = 11;
const SR2: i32 = 1;

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
unsafe fn sr_128_lane_sse(x: __m128i) -> __m128i {
    let part1 = _mm_srli_epi32(x, SR2);
    let tmp = _mm_srli_si128(x, 4);
    let part2 = _mm_slli_epi32(tmp, 32 - SR2);

    _mm_or_si128(part1, part2)
}

//
// shift left
//

const SL1: i32 = 18;
const SL2: i32 = 1;

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
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sl_128_lane_sse(x: __m128i) -> __m128i {
    let part1 = _mm_slli_epi32(x, SL2);
    let tmp = _mm_slli_si128(x, 4);
    let part2 = _mm_srli_epi32(tmp, 32 - SL2);

    _mm_or_si128(part1, part2)
}
