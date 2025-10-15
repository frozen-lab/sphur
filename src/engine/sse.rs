//! # SSE Engine
//!
//! **Only for x86_64 architectures!**
//!
//! This module implements the SFMT algorithm using the **SSE2** instruction set (and compatible
//! upgraded ISA variants).
//!
//! This impl is **fully inlined**, **branch-minimized**, and manually unrolled for
//! **ILP (Instruction-Level Parallelism)**!
//!
//! CAUTION: Handle with care, this is the hot path 😜.
use core::arch::x86_64::*;

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
        let pos1 = POS1;
        let ptr = state.as_mut_ptr();

        // precompute for commy offsets
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

            // ILP optimized compute
            let out0 = recurrence_relation(a_0, b_0, c_0, d_0);
            let out1 = recurrence_relation(a_1, b_1, c_1, d_1);

            // write back
            core::ptr::write(ptr.add(i0), out0);
            core::ptr::write(ptr.add(i1), out1);

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
    let t0 = _mm_xor_si128(a, _mm_slli_epi32(a, SL1 as i32));

    // by = elem's shift + mask
    let by = _mm_and_si128(_mm_srli_epi32(b, SR1 as i32), MASK128);

    // t1 = combine c and d shifts
    let c_sr2 = sr_128_lane_sse(c);
    let d_sl2 = sl_128_lane_sse(d);
    let t1 = _mm_xor_si128(c_sr2, d_sl2);

    // out = ((a ^ ax) ^ by) ^ (c_sr2 ^ d_sl2)
    let left = _mm_xor_si128(by, t0);
    _mm_xor_si128(t1, left)
}

#[cfg(test)]
mod sse {
    use super::*;
    use crate::engine::Engine;

    #[cfg(test)]
    mod engine {
        use super::*;

        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn lane_to_u32s(v: __m128i) -> [u32; 4] {
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v);

            out
        }

        mod init {
            use super::*;

            #[test]
            fn test_new_deterministic_same_seed() {
                unsafe {
                    let s1 = SSE::new(0xDEADBEEFu64);
                    let s2 = SSE::new(0xDEADBEEFu64);

                    let v1: Vec<[u32; 4]> = s1.iter().map(|&l| lane_to_u32s(l)).collect();
                    let v2: Vec<[u32; 4]> = s2.iter().map(|&l| lane_to_u32s(l)).collect();

                    assert_eq!(v1, v2, "engine::new must be deterministic for identical seeds");
                }
            }

            #[test]
            fn test_new_distinct_for_different_seeds() {
                unsafe {
                    let s1 = SSE::new(0x1111u64);
                    let s2 = SSE::new(0x2222u64);

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
                    let state = SSE::new(0xBEEFu64);

                    // length sanity
                    assert_eq!(state.len(), SSE_STATE_LEN);

                    // __m128i must be 16-byte aligned
                    let ptr = state.as_ptr() as usize;
                    assert_eq!(ptr % 16, 0, "state base pointer must be 16-byte aligned for SSE");

                    // ensure state is not trivially all-zero
                    let all_zero = state.iter().all(|&l| lane_to_u32s(l) == [0u32; 4]);
                    assert!(!all_zero, "initialized state should not be all zeros");
                }
            }

            #[test]
            fn test_new_small_seed_variation() {
                unsafe {
                    // ensure small changes in seed change state (sanity)
                    let s1 = SSE::new(0x1000u64);
                    let s2 = SSE::new(0x1002u64);

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
            unsafe fn state_to_vec(state: &[__m128i; SSE_STATE_LEN]) -> Vec<[u32; 4]> {
                state.iter().map(|&l| lane_to_u32s(l)).collect()
            }

            #[test]
            fn test_regen_deterministic_and_changes() {
                unsafe {
                    let mut s1 = SSE::new(0xDEADBEEFu64);
                    let mut s2 = SSE::new(0xDEADBEEFu64);

                    // og snap
                    let og = state_to_vec(&s1);

                    // regen one copy
                    SSE::regen(&mut s1);

                    // regen the other copy the same way, should match
                    SSE::regen(&mut s2);

                    let after1 = state_to_vec(&s1);
                    let after2 = state_to_vec(&s2);

                    assert_eq!(after1, after2, "regen must be deterministic");
                    assert_ne!(og, after1, "regen should modify the state");
                }
            }

            #[test]
            fn test_regen_multi_step_no_short_cycle() {
                unsafe {
                    let mut state = SSE::new(0xCAFEBABEu64);
                    let mut seen: HashSet<Vec<[u32; 4]>> = HashSet::new();

                    // record initial
                    seen.insert(state_to_vec(&state));

                    // run several regenerations and assert we don't repeat within this window
                    for _ in 0..20 {
                        SSE::regen(&mut state);
                        let v = state_to_vec(&state);
                        assert!(
                            !seen.contains(&v),
                            "regen produced a repeat state within 20 steps (very unlikely)"
                        );
                        seen.insert(v);
                    }
                }
            }

            #[test]
            fn test_regen_wraparound_sanity() {
                unsafe {
                    // fill state with distinctive lane values equal to index (so we can see change)
                    let mut state: [__m128i; SSE_STATE_LEN] =
                        core::array::from_fn(|i| _mm_set_epi32(i as i32, i as i32, i as i32, i as i32));

                    // snapshot a few end indices
                    let n = SSE_STATE_LEN;
                    let idxs = [n - 3, n - 2, n - 1];

                    let before: Vec<[u32; 4]> = idxs.iter().map(|&i| lane_to_u32s(state[i])).collect();

                    // run regen (should handle wraparound without panic)
                    SSE::regen(&mut state);

                    let after: Vec<[u32; 4]> = idxs.iter().map(|&i| lane_to_u32s(state[i])).collect();

                    // ensure at least one of the tail lanes changed (sanity for wrap logic)
                    assert!(
                        before != after,
                        "expected at least one wrapped index to change after regen"
                    );
                }
            }
        }
    }

    mod gen_functions {
        use super::*;

        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn set_lane_u32s(vals: [u32; 4]) -> __m128i {
            _mm_set_epi32(vals[3] as i32, vals[2] as i32, vals[1] as i32, vals[0] as i32)
        }

        #[test]
        fn test_batch_u32_and_u64_consistency() {
            unsafe {
                let state = [set_lane_u32s([1, 2, 3, 4]); SSE_STATE_LEN];
                let lane_idx = 0;

                let got32 = SSE::batch_u32(&state, lane_idx);
                let got64 = SSE::batch_u64(&state, lane_idx);

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
                let state = [lane; SSE_STATE_LEN];

                let out = SSE::batch_u32(&state, 0);
                assert_eq!(out, vals);
            }
        }

        #[test]
        fn test_batch_u64_correct_order() {
            unsafe {
                // 4 u32s => 2 u64s (little-endian order)
                let vals = [0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00];
                let lane = set_lane_u32s(vals);
                let state = [lane; SSE_STATE_LEN];

                // interpret little-endian pairs
                let out64 = SSE::batch_u64(&state, 0);
                let low = (vals[1] as u64) << 32 | vals[0] as u64;
                let high = (vals[3] as u64) << 32 | vals[2] as u64;

                assert_eq!(out64, [low, high]);
            }
        }

        #[test]
        fn test_batch_bounds_assertion() {
            unsafe {
                let state = [_mm_setzero_si128(); SSE_STATE_LEN];
                let last_ok = SSE::batch_u32(&state, SSE_STATE_LEN - 1);

                assert_eq!(last_ok, [0; 4]);
            }
        }
    }

    mod indep_functions {
        use super::*;

        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn to_u32s(v: __m128i) -> [u32; 4] {
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v);

            out
        }

        mod sl_128_lane {
            use super::*;

            #[test]
            fn test_sl_128_basic_pattern() {
                unsafe {
                    let x = _mm_set_epi32(0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d);
                    let y = sl_128_lane_sse(x);
                    let got = to_u32s(y);

                    assert_ne!(got, [0; 4]);
                }
            }

            #[test]
            fn test_sl_128_all_zeros() {
                unsafe {
                    let x = _mm_set1_epi32(0);
                    let y = sl_128_lane_sse(x);

                    assert_eq!(to_u32s(y), [0; 4]);
                }
            }

            #[test]
            fn test_sl_128_all_ones() {
                unsafe {
                    let x = _mm_set1_epi32(u32::MAX as i32);
                    let y = sl_128_lane_sse(x);

                    assert_eq!(to_u32s(y), [0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]);
                }
            }
        }

        mod sr_128_lane {
            use super::*;

            #[test]
            fn test_sr_128_basic_pattern() {
                unsafe {
                    let x = _mm_set_epi32(0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d);
                    let y = sr_128_lane_sse(x);
                    let got = to_u32s(y);

                    assert_ne!(got, [0; 4]);
                }
            }

            #[test]
            fn test_sr_128_all_zeros() {
                unsafe {
                    let x = _mm_set1_epi32(0);
                    let y = sr_128_lane_sse(x);

                    assert_eq!(to_u32s(y), [0; 4]);
                }
            }

            #[test]
            fn test_sr_128_all_ones() {
                unsafe {
                    let x = _mm_set1_epi32(u32::MAX as i32);
                    let y = sr_128_lane_sse(x);

                    assert_eq!(to_u32s(y), [u32::MAX, u32::MAX, u32::MAX, 0x7FFF_FFFF]);
                }
            }
        }

        mod recurrence_relation {
            use super::*;

            #[test]
            fn test_recurrence_deterministic_known_inputs() {
                unsafe {
                    let a = _mm_set_epi32(1, 2, 3, 4);
                    let b = _mm_set_epi32(5, 6, 7, 8);
                    let c = _mm_set_epi32(9, 10, 11, 12);
                    let d = _mm_set_epi32(13, 14, 15, 16);

                    let out = recurrence_relation(a, b, c, d);
                    let got = to_u32s(out);

                    assert_ne!(got, [0; 4]);
                    assert_eq!(to_u32s(out), to_u32s(recurrence_relation(a, b, c, d)));
                }
            }

            #[test]
            fn test_masking_effect_visible() {
                unsafe {
                    let a = _mm_set1_epi32(0xAAAAAAAAu32 as i32);
                    let b = _mm_set1_epi32(0xFFFFFFFFu32 as i32);
                    let c = _mm_set1_epi32(0x55555555u32 as i32);
                    let d = _mm_set1_epi32(0x12345678u32 as i32);

                    let out_full = recurrence_relation(a, b, c, d);

                    // masking b with zeros should change output
                    let b_zero = _mm_set1_epi32(0);
                    let out_masked = recurrence_relation(a, b_zero, c, d);

                    assert_ne!(to_u32s(out_full), to_u32s(out_masked));
                }
            }

            #[test]
            fn test_inputs_unchanged() {
                unsafe {
                    let a = _mm_set_epi32(1, 2, 3, 4);
                    let b = _mm_set_epi32(5, 6, 7, 8);
                    let c = _mm_set_epi32(9, 10, 11, 12);
                    let d = _mm_set_epi32(13, 14, 15, 16);

                    let orig_a = to_u32s(a);
                    let orig_b = to_u32s(b);
                    let orig_c = to_u32s(c);
                    let orig_d = to_u32s(d);

                    let _ = recurrence_relation(a, b, c, d);

                    assert_eq!(to_u32s(a), orig_a);
                    assert_eq!(to_u32s(b), orig_b);
                    assert_eq!(to_u32s(c), orig_c);
                    assert_eq!(to_u32s(d), orig_d);
                }
            }

            #[test]
            fn test_symmetry_breaks_if_swapped() {
                unsafe {
                    let a = _mm_set_epi32(1, 2, 3, 4);
                    let b = _mm_set_epi32(5, 6, 7, 8);
                    let c = _mm_set_epi32(9, 10, 11, 12);
                    let d = _mm_set_epi32(13, 14, 15, 16);

                    let out1 = recurrence_relation(a, b, c, d);
                    let out2 = recurrence_relation(d, c, b, a);

                    assert_ne!(to_u32s(out1), to_u32s(out2));
                }
            }
        }
    }
}
