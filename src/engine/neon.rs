//! # NEON Engine
//!
//! **Only for aarch64 architectures!**
//!
//! This module implements the SFMT algorithm using the **NEON** instruction set.
//!
//! This impl is **fully inlined**, **branch-minimized**, and manually unrolled for
//! **ILP (Instruction-Level Parallelism)**!
//!
use core::arch::aarch64::*;
use core::mem::transmute;

pub(crate) const NEON_STATE_LEN: usize = 156;
pub(crate) const NEON_N64: usize = 2;
pub(crate) const NEON_N32: usize = 4;

const _: () = assert!(NEON_STATE_LEN % 2 == 0);
const _: () = assert!(NEON_N64 % 2 == 0);
const _: () = assert!(NEON_N32 % 2 == 0);

const POS1: usize = 122;

const SR1: i32 = 11;
const SR2: i32 = 1;
const SL1: i32 = 18;
const SL2: i32 = 1;

const MASK_WORDS_U32: [u32; 4] = [
    super::MASK_WORDS[0] as u32,
    super::MASK_WORDS[1] as u32,
    super::MASK_WORDS[2] as u32,
    super::MASK_WORDS[3] as u32,
];

pub(crate) struct NEON;

impl super::Engine<NEON_STATE_LEN, NEON_N64, NEON_N32> for NEON {
    type Lane = uint32x4_t;

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn new(seed: u64) -> [Self::Lane; NEON_STATE_LEN] {
        super::init_engine_state::<Self::Lane, NEON_STATE_LEN, { NEON_STATE_LEN * 4 }>(seed, |w| vld1q_u32(w.as_ptr()))
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn regen(state: &mut [Self::Lane; NEON_STATE_LEN]) {
        let n = NEON_STATE_LEN;
        let pos1 = POS1;
        let ptr = state.as_mut_ptr();
        let mask = vld1q_u32(MASK_WORDS_U32.as_ptr());

        let n_minus_2 = n - 2;
        let n_minus_1 = n - 1;

        let mut i0 = 0usize;

        while i0 + 1 < n {
            let b0 = {
                let t = i0 + pos1;

                if t >= n {
                    t - n
                } else {
                    t
                }
            };

            let c0 = if i0 + n_minus_2 >= n {
                i0 + n_minus_2 - n
            } else {
                i0 + n_minus_2
            };

            let d0 = if i0 + n_minus_1 >= n {
                i0 + n_minus_1 - n
            } else {
                i0 + n_minus_1
            };

            let i1 = i0 + 1;

            let b1 = {
                let t = i1 + pos1;

                if t >= n {
                    t - n
                } else {
                    t
                }
            };

            let c1 = if i1 + n_minus_2 >= n {
                i1 + n_minus_2 - n
            } else {
                i1 + n_minus_2
            };

            let d1 = if i1 + n_minus_1 >= n {
                i1 + n_minus_1 - n
            } else {
                i1 + n_minus_1
            };

            let a_0 = core::ptr::read(ptr.add(i0));
            let b_0 = core::ptr::read(ptr.add(b0));
            let c_0 = core::ptr::read(ptr.add(c0));
            let d_0 = core::ptr::read(ptr.add(d0));

            let a_1 = core::ptr::read(ptr.add(i1));
            let b_1 = core::ptr::read(ptr.add(b1));
            let c_1 = core::ptr::read(ptr.add(c1));
            let d_1 = core::ptr::read(ptr.add(d1));

            let out0 = recurrence_relation(a_0, b_0, c_0, d_0, mask);
            let out1 = recurrence_relation(a_1, b_1, c_1, d_1, mask);

            core::ptr::write(ptr.add(i0), out0);
            core::ptr::write(ptr.add(i1), out1);

            i0 += 2;
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn batch_u64(state: &[Self::Lane; NEON_STATE_LEN], lane: usize) -> [u64; NEON_N64] {
        debug_assert!(lane < NEON_STATE_LEN, "Lane must be in bounds with the state size");

        transmute::<uint32x4_t, [u64; NEON_N64]>(*state.get_unchecked(lane))
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn batch_u32(state: &[Self::Lane; NEON_STATE_LEN], lane: usize) -> [u32; NEON_N32] {
        debug_assert!(lane < NEON_STATE_LEN, "Lane must be in bounds with the state size");

        transmute::<uint32x4_t, [u32; NEON_N32]>(*state.get_unchecked(lane))
    }
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sr_128_lane_neon(x: uint32x4_t) -> uint32x4_t {
    let part1 = vshrq_n_u32(x, SR2 as u32);

    let x_u8 = vreinterpretq_u8_u32(x);
    let zero_u8 = vdupq_n_u8(0);

    let tmp_bytes = vextq_u8(zero_u8, x_u8, 12);
    let tmp_u32 = vreinterpretq_u32_u8(tmp_bytes);

    let part2 = vshlq_n_u32(tmp_u32, (32 - SR2) as u32);

    vorrq_u32(part1, part2)
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sl_128_lane_neon(x: uint32x4_t) -> uint32x4_t {
    let part1 = vshlq_n_u32(x, SL2 as u32);

    let x_u8 = vreinterpretq_u8_u32(x);
    let zero_u8 = vdupq_n_u8(0);

    let tmp_bytes = vextq_u8(x_u8, zero_u8, 4);
    let tmp_u32 = vreinterpretq_u32_u8(tmp_bytes);

    let part2 = vshrq_n_u32(tmp_u32, (32 - SL2) as u32);

    vorrq_u32(part1, part2)
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn recurrence_relation(
    a: uint32x4_t,
    b: uint32x4_t,
    c: uint32x4_t,
    d: uint32x4_t,
    mask: uint32x4_t,
) -> uint32x4_t {
    // t0 = a ^ (a << SL1)
    let t0 = veorq_u32(a, vshlq_n_u32(a, SL1 as u32));

    // by = (b >> SR1) & mask
    let by = vandq_u32(vshrq_n_u32(b, SR1 as u32), mask);

    // t1 = sr128(c) ^ sl128(d)
    let c_sr2 = sr_128_lane_neon(c);
    let d_sl2 = sl_128_lane_neon(d);
    let t1 = veorq_u32(c_sr2, d_sl2);

    // out = t0 ^ by ^ t1 (same grouping/order as SSE impl to preserve ILP)
    let left = veorq_u32(by, t0);
    veorq_u32(t1, left)
}
