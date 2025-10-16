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

const MASK_WORDS: [u32; 4] = [0xbffffff6u32, 0xbffaffffu32, 0xddfecb7fu32, 0xdfffffefu32];

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
        let mask = vld1q_u32(MASK_WORDS.as_ptr());

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
    let part1 = vshrq_n_u32(x, SR2 as i32);

    let x_u8 = vreinterpretq_u8_u32(x);
    let zero_u8 = vdupq_n_u8(0);

    let tmp_bytes = vextq_u8(zero_u8, x_u8, 12);
    let tmp_u32 = vreinterpretq_u32_u8(tmp_bytes);

    let part2 = vshlq_n_u32(tmp_u32, (32 - SR2) as i32);

    vorrq_u32(part1, part2)
}

#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sl_128_lane_neon(x: uint32x4_t) -> uint32x4_t {
    let part1 = vshlq_n_u32(x, SL2 as i32);

    let x_u8 = vreinterpretq_u8_u32(x);
    let zero_u8 = vdupq_n_u8(0);

    let tmp_bytes = vextq_u8(x_u8, zero_u8, 4);
    let tmp_u32 = vreinterpretq_u32_u8(tmp_bytes);

    let part2 = vshrq_n_u32(tmp_u32, (32 - SL2) as i32);

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
    let t0 = veorq_u32(a, vshlq_n_u32(a, SL1 as i32));

    // by = (b >> SR1) & mask
    let by = vandq_u32(vshrq_n_u32(b, SR1 as i32), mask);

    // t1 = sr128(c) ^ sl128(d)
    let c_sr2 = sr_128_lane_neon(c);
    let d_sl2 = sl_128_lane_neon(d);
    let t1 = veorq_u32(c_sr2, d_sl2);

    // out = t0 ^ by ^ t1 (same grouping/order as SSE impl to preserve ILP)
    let left = veorq_u32(by, t0);
    veorq_u32(t1, left)
}

#[cfg(test)]
mod neon {
    use super::*;
    use crate::engine::Engine;

    #[cfg(test)]
    mod engine {
        use super::*;

        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn lane_to_u32s(v: uint32x4_t) -> [u32; 4] {
            let mut out = [0u32; 4];
            vst1q_u32(out.as_mut_ptr(), v);
            out
        }

        mod init {
            use super::*;

            #[test]
            fn test_new_deterministic_same_seed() {
                unsafe {
                    let s1 = NEON::new(0xDEADBEEFu64);
                    let s2 = NEON::new(0xDEADBEEFu64);

                    let v1: Vec<[u32; 4]> = s1.iter().map(|&l| lane_to_u32s(l)).collect();
                    let v2: Vec<[u32; 4]> = s2.iter().map(|&l| lane_to_u32s(l)).collect();

                    assert_eq!(v1, v2, "engine::new must be deterministic for identical seeds");
                }
            }

            #[test]
            fn test_new_distinct_for_different_seeds() {
                unsafe {
                    let s1 = NEON::new(0x1111u64);
                    let s2 = NEON::new(0x2222u64);

                    let v1: Vec<[u32; 4]> = s1.iter().map(|&l| lane_to_u32s(l)).collect();
                    let v2: Vec<[u32; 4]> = s2.iter().map(|&l| lane_to_u32s(l)).collect();

                    assert_ne!(
                        v1, v2,
                        "different seeds should produce different initial states (very likely)"
                    );
                }
            }

            #[test]
            fn test_new_alignment_and_length() {
                unsafe {
                    let state = NEON::new(0xBEEFu64);
                    assert_eq!(state.len(), NEON_STATE_LEN);

                    let ptr = state.as_ptr() as usize;
                    assert_eq!(ptr % 16, 0, "state must be 16-byte aligned for NEON");

                    let all_zero = state.iter().all(|&l| lane_to_u32s(l) == [0u32; 4]);
                    assert!(!all_zero, "initialized state should not be all zeros");
                }
            }

            #[test]
            fn test_new_small_seed_variation() {
                unsafe {
                    let s1 = NEON::new(0x1000u64);
                    let s2 = NEON::new(0x1002u64);

                    let v1: Vec<[u32; 4]> = s1.iter().map(|&l| lane_to_u32s(l)).collect();
                    let v2: Vec<[u32; 4]> = s2.iter().map(|&l| lane_to_u32s(l)).collect();

                    assert_ne!(v1, v2, "adjacent seeds should produce different states");
                }
            }
        }

        mod regen {
            use super::*;
            use std::collections::HashSet;

            #[allow(unsafe_op_in_unsafe_fn)]
            unsafe fn state_to_vec(state: &[uint32x4_t; NEON_STATE_LEN]) -> Vec<[u32; 4]> {
                state.iter().map(|&l| lane_to_u32s(l)).collect()
            }

            #[test]
            fn test_regen_deterministic_and_changes() {
                unsafe {
                    let mut s1 = NEON::new(0xDEADBEEFu64);
                    let mut s2 = NEON::new(0xDEADBEEFu64);

                    let og = state_to_vec(&s1);

                    NEON::regen(&mut s1);
                    NEON::regen(&mut s2);

                    let after1 = state_to_vec(&s1);
                    let after2 = state_to_vec(&s2);

                    assert_eq!(after1, after2, "regen must be deterministic");
                    assert_ne!(og, after1, "regen should modify state");
                }
            }

            #[test]
            fn test_regen_multi_step_no_short_cycle() {
                unsafe {
                    let mut state = NEON::new(0xCAFEBABEu64);
                    let mut seen: HashSet<Vec<[u32; 4]>> = HashSet::new();

                    seen.insert(state_to_vec(&state));

                    for _ in 0..20 {
                        NEON::regen(&mut state);
                        let v = state_to_vec(&state);
                        assert!(!seen.contains(&v), "regen produced a repeat within 20 steps (unlikely)");
                        seen.insert(v);
                    }
                }
            }

            #[test]
            fn test_regen_wraparound_sanity() {
                unsafe {
                    let mut state: [uint32x4_t; NEON_STATE_LEN] = core::array::from_fn(|i| vdupq_n_u32(i as u32));

                    let n = NEON_STATE_LEN;
                    let idxs = [n - 3, n - 2, n - 1];
                    let before: Vec<[u32; 4]> = idxs.iter().map(|&i| lane_to_u32s(state[i])).collect();

                    NEON::regen(&mut state);

                    let after: Vec<[u32; 4]> = idxs.iter().map(|&i| lane_to_u32s(state[i])).collect();

                    assert_ne!(before, after, "expected tail indices to change after regen");
                }
            }
        }
    }

    mod gen_functions {
        use super::*;

        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn set_lane_u32s(vals: [u32; 4]) -> uint32x4_t {
            vld1q_u32(vals.as_ptr())
        }

        #[test]
        fn test_batch_u32_and_u64_consistency() {
            unsafe {
                let state = [set_lane_u32s([1, 2, 3, 4]); NEON_STATE_LEN];
                let lane_idx = 0;

                let got32 = NEON::batch_u32(&state, lane_idx);
                let got64 = NEON::batch_u64(&state, lane_idx);

                let bytes32: [u8; 16] = core::mem::transmute(got32);
                let bytes64: [u8; 16] = core::mem::transmute(got64);

                assert_eq!(bytes32, bytes64);
            }
        }

        #[test]
        fn test_batch_u32_basic_values() {
            unsafe {
                let vals = [10, 20, 30, 40];
                let lane = set_lane_u32s(vals);
                let state = [lane; NEON_STATE_LEN];

                let out = NEON::batch_u32(&state, 0);
                assert_eq!(out, vals);
            }
        }

        #[test]
        fn test_batch_u64_correct_order() {
            unsafe {
                let vals = [0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00];
                let lane = set_lane_u32s(vals);
                let state = [lane; NEON_STATE_LEN];

                let out64 = NEON::batch_u64(&state, 0);
                let low = (vals[1] as u64) << 32 | vals[0] as u64;
                let high = (vals[3] as u64) << 32 | vals[2] as u64;

                assert_eq!(out64, [low, high]);
            }
        }
    }

    mod indep_functions {
        use super::*;

        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn to_u32s(v: uint32x4_t) -> [u32; 4] {
            let mut out = [0u32; 4];
            vst1q_u32(out.as_mut_ptr(), v);
            out
        }

        mod sl_128_lane {
            use super::*;

            #[test]
            fn test_sl_128_basic_pattern() {
                unsafe {
                    let x = vld1q_u32([1, 2, 3, 4].as_ptr());
                    let y = sl_128_lane_neon(x);
                    let got = to_u32s(y);
                    assert_ne!(got, [0; 4]);
                }
            }

            #[test]
            fn test_sl_128_all_zeros() {
                unsafe {
                    let x = vdupq_n_u32(0);
                    let y = sl_128_lane_neon(x);
                    assert_eq!(to_u32s(y), [0; 4]);
                }
            }
        }

        mod sr_128_lane {
            use super::*;

            #[test]
            fn test_sr_128_basic_pattern() {
                unsafe {
                    let x = vld1q_u32([1, 2, 3, 4].as_ptr());
                    let y = sr_128_lane_neon(x);
                    assert_ne!(to_u32s(y), [0; 4]);
                }
            }

            #[test]
            fn test_sr_128_all_zeros() {
                unsafe {
                    let x = vdupq_n_u32(0);
                    let y = sr_128_lane_neon(x);
                    assert_eq!(to_u32s(y), [0; 4]);
                }
            }
        }

        mod recurrence_relation {
            use super::*;

            fn mask() -> uint32x4_t {
                unsafe { vld1q_u32(MASK_WORDS.as_ptr()) }
            }

            #[test]
            fn test_recurrence_deterministic_known_inputs() {
                unsafe {
                    let a = vld1q_u32([1, 2, 3, 4].as_ptr());
                    let b = vld1q_u32([5, 6, 7, 8].as_ptr());
                    let c = vld1q_u32([9, 10, 11, 12].as_ptr());
                    let d = vld1q_u32([13, 14, 15, 16].as_ptr());

                    let out = recurrence_relation(a, b, c, d, mask());
                    let got = to_u32s(out);

                    assert_ne!(got, [0; 4]);
                    assert_eq!(to_u32s(out), to_u32s(recurrence_relation(a, b, c, d, mask())));
                }
            }

            #[test]
            fn test_masking_effect_visible() {
                unsafe {
                    let a = vdupq_n_u32(0xAAAAAAAA);
                    let b = vdupq_n_u32(0xFFFFFFFF);
                    let c = vdupq_n_u32(0x55555555);
                    let d = vdupq_n_u32(0x12345678);

                    let out_full = recurrence_relation(a, b, c, d, mask());
                    let b_zero = vdupq_n_u32(0);
                    let out_masked = recurrence_relation(a, b_zero, c, d, mask());

                    assert_ne!(to_u32s(out_full), to_u32s(out_masked));
                }
            }
        }
    }
}
