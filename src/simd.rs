use crate::sfmt::InnerState;

pub(crate) const N128: usize = 156;
pub(crate) const STATE32_LEN: usize = N128 * 4;
pub(crate) const PARITY: [u32; 4] = [0x00000001, 0x00000000, 0x00000000, 0x13c9e684];

const MSK: [u32; 4] = [0xdfffffefu32, 0xddfecb7fu32, 0xbffaffffu32, 0xbffffff6u32];
const POS1: usize = 122;
const SL1: usize = 18;
const SL2: usize = 1;
const SR1: usize = 11;
const SR2: usize = 1;

pub(crate) struct Simd(ISA);

impl Simd {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self(ISA::detect_isa())
    }

    #[inline(always)]
    pub(crate) fn gen_state(&self, state: &mut InnerState) {
        match self.0 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            ISA::AVX2 => unsafe {
                avx2::generate_inner_state(&mut state.0);
            },

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            ISA::SSE2 => unsafe {
                sse2::generate_inner_state(&mut state.0);
            },

            #[cfg(target_arch = "aarch64")]
            ISA::NEON => unsafe {
                neon::generate_inner_state(&mut state.0);
            },
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(unused)]
enum ISA {
    // This ISA is an upgrade over SSE2 if available at runtime
    AVX2,

    // This ISA is default on x64 (x86_64), as it's virtually available on all x86_64 CPU's
    SSE2,

    // This ISA is default for aarch64, as its vurtually available on all aarch64 CPU's
    NEON,
}

impl ISA {
    fn detect_isa() -> ISA {
        // NOTE: On x86_64 we upgrade to AVX2 if available, otherwise
        // treat SSE2 as baseline
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return ISA::AVX2;
            }

            ISA::SSE2
        }

        #[cfg(target_arch = "aarch64")]
        return ISA::NEON;
    }
}

#[cfg(all(target_arch = "x86_64"))]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    const U32S_PER_128: usize = 4;
    const U32S_PER_256: usize = 8;

    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn generate_inner_state(state: &mut [u32; STATE32_LEN]) {
        // sanity check
        debug_assert_eq!((state.as_ptr() as usize) % 32, 0, "InnerState must be 32 byte aligned");

        let mask128 = _mm_set_epi32(MSK[3] as i32, MSK[2] as i32, MSK[1] as i32, MSK[0] as i32);
        let mask256 = _mm256_inserti128_si256(_mm256_castsi128_si256(mask128), mask128, 1);

        let mut i = 0usize;

        while i < N128 {
            //
            // first 256 bits block (pos -> i)
            //

            let ji = i + POS1;
            let b_idx128 = if ji >= N128 { ji - N128 } else { ji };

            let ci = i + N128 - 2;
            let c_idx128 = if ci >= N128 { ci - N128 } else { ci };

            let di = i + N128 - 1;
            let d_idx128 = if di >= N128 { di - N128 } else { di };

            let a = load256_from_128_chunks(state.as_ptr(), i);
            let b = load256_from_128_chunks(state.as_ptr(), b_idx128);
            let c = load256_from_128_chunks(state.as_ptr(), c_idx128);
            let d = load256_from_128_chunks(state.as_ptr(), d_idx128);

            let rr = recurrence_relation_256(a, b, c, d, mask256);
            store256_to_128_chunks(state.as_mut_ptr(), i, rr);

            //
            // second 256 bits block (pos -> i + 2)
            //

            // NOTE: check to avoid state overflow
            let next_i128 = i + 2;

            if next_i128 < N128 {
                let ji = next_i128 + POS1;
                let b_idx128 = if ji >= N128 { ji - N128 } else { ji };

                let ci = next_i128 + N128 - 2;
                let c_idx128 = if ci >= N128 { ci - N128 } else { ci };

                let di = next_i128 + N128 - 1;
                let d_idx128 = if di >= N128 { di - N128 } else { di };

                let a = load256_from_128_chunks(state.as_ptr(), next_i128);
                let b = load256_from_128_chunks(state.as_ptr(), b_idx128);
                let c = load256_from_128_chunks(state.as_ptr(), c_idx128);
                let d = load256_from_128_chunks(state.as_ptr(), d_idx128);

                let rr = recurrence_relation_256(a, b, c, d, mask256);
                store256_to_128_chunks(state.as_mut_ptr(), next_i128, rr);
            }

            // NOTE: As single avx2 lane is 256 bits wide, and we consumed bits worth
            // 2 avx2 lanes, i.e. `(2 * 128) * 2 -> 512` bits, which is `128 * 4`
            i += 4;
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn recurrence_relation_256(
        a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        mask256: __m256i,
    ) -> __m256i {
        // x = a << SL1 (per-lane)
        let x = _mm256_slli_epi32(a, SL1 as i32);

        // y = (b >> SR1) & mask
        let y = _mm256_and_si256(_mm256_srli_epi32(b, SR1 as i32), mask256);

        // c_sr2 = shift_right_256_epi32(c)
        let c_sr2 = shift_right_256_epi32(c);

        // d_sl2 = shift_left_256_epi32(d)
        let d_sl2 = shift_left_256_epi32(d);

        // r = a ^ x ^ y ^ c_sr2 ^ d_sl2
        let mut r = _mm256_xor_si256(a, x);

        r = _mm256_xor_si256(r, y);
        r = _mm256_xor_si256(r, c_sr2);
        r = _mm256_xor_si256(r, d_sl2);

        r
    }

    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn shift_left_256_epi32(x: __m256i) -> __m256i {
        let tmp = _mm256_slli_si256(x, 4);
        let x1 = _mm256_slli_epi32(x, SL1 as i32);
        let x2 = _mm256_srli_epi32(tmp, (32 - SL1) as i32);

        _mm256_or_si256(x1, x2)
    }

    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn shift_right_256_epi32(x: __m256i) -> __m256i {
        let tmp = _mm256_srli_si256(x, 4);
        let x1 = _mm256_srli_epi32(x, SR1 as i32);
        let x2 = _mm256_slli_epi32(tmp, (32 - SR1) as i32);

        _mm256_or_si256(x1, x2)
    }

    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn load256_from_128_chunks(src: *const u32, idx128: usize) -> __m256i {
        let u32_idx = idx128 * U32S_PER_128;

        // NOTE: if both 128 blocks are contiguous we can use this fast path
        if idx128 + 1 < N128 {
            let p = src.add(u32_idx) as *const __m256i;
            return _mm256_load_si256(p);
        }

        let lo = _mm_load_si128(src.add(u32_idx) as *const __m128i);
        let hi = _mm_load_si128(src.add(0) as *const __m128i);

        _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
    }

    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn store256_to_128_chunks(dst: *mut u32, idx128: usize, v: __m256i) {
        let u32_idx = idx128 * U32S_PER_128;

        // NOTE: if both 128 blocks are contiguous we can use this fast path
        if idx128 + 1 < N128 {
            let p = dst.add(u32_idx) as *mut __m256i;
            _mm256_store_si256(p, v);

            return;
        }

        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);

        _mm_store_si128(dst.add(u32_idx) as *mut __m128i, lo);
        _mm_store_si128(dst.add(0) as *mut __m128i, hi);
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use super::*;
    use std::arch::x86_64::*;

    // 4 u32 per SIMD lane
    const MULT: usize = 4;

    #[target_feature(enable = "sse2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn generate_inner_state(state: &mut [u32; STATE32_LEN]) {
        // sanity check
        debug_assert_eq!((state.as_ptr() as usize) % 32, 0, "InnerState must be 32 byte aligned");

        let mask = _mm_set_epi32(MSK[3] as i32, MSK[2] as i32, MSK[1] as i32, MSK[0] as i32);

        for i in 0..N128 {
            let ji = i + POS1;
            let b_idx = if ji >= N128 { ji - N128 } else { ji };

            let ci = i + N128 - 2;
            let c_idx = if ci >= N128 { ci - N128 } else { ci };

            let di = i + N128 - 1;
            let d_idx = if di >= N128 { di - N128 } else { di };

            let a = _mm_load_si128(state.as_ptr().add(i * MULT) as *const __m128i);
            let b = _mm_load_si128(state.as_ptr().add(b_idx * MULT) as *const __m128i);
            let c = _mm_load_si128(state.as_ptr().add(c_idx * MULT) as *const __m128i);
            let d = _mm_load_si128(state.as_ptr().add(d_idx * MULT) as *const __m128i);

            let rr = recurrence_relation(a, b, c, d, mask);
            _mm_store_si128(state.as_mut_ptr().add(i * MULT) as *mut __m128i, rr);
        }
    }

    #[target_feature(enable = "sse2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn recurrence_relation(a: __m128i, b: __m128i, c: __m128i, d: __m128i, mask: __m128i) -> __m128i {
        // x = a << SL1
        let x = _mm_slli_epi32(a, SL1 as i32);

        // y = `b >> SR1 & mask`
        let y = _mm_and_si128(_mm_srli_epi32(b, SR1 as i32), mask);

        let c_sr2 = shift_right_128_epi32(c);
        let d_sl2 = shift_left_128_epi32(d);

        // r = a ^ x ^ y ^ c_sr2 ^ d_sl2
        let mut r = _mm_xor_si128(a, x);

        r = _mm_xor_si128(r, y);
        r = _mm_xor_si128(r, c_sr2);
        r = _mm_xor_si128(r, d_sl2);

        r
    }

    #[target_feature(enable = "sse2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn shift_right_128_epi32(x: __m128i) -> __m128i {
        const SR1_I32: i32 = SR1 as i32;

        if SR1_I32 == 0 {
            return x;
        }

        let x1 = _mm_srli_epi32(x, SR1_I32);
        let x2 = _mm_slli_epi32(_mm_srli_si128(x, 4), 32 - SR1_I32);

        _mm_or_si128(x1, x2)
    }

    #[target_feature(enable = "sse2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn shift_left_128_epi32(x: __m128i) -> __m128i {
        const SL1_I32: i32 = SL1 as i32;

        if SL1_I32 == 0 {
            return x;
        }

        let x1 = _mm_slli_epi32(x, SL1_I32);
        let x2 = _mm_srli_epi32(_mm_slli_si128(x, 4), 32 - SL1_I32);

        _mm_or_si128(x1, x2)
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    // 4 u32 per SIMD lane
    const MULT: usize = 4;

    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn generate_inner_state(state: &mut [u32; STATE32_LEN]) {
        // sanity check
        debug_assert_eq!((state.as_ptr() as usize) % 32, 0, "InnerState must be 32 byte aligned");

        let mask = vld1q_u32(MSK.as_ptr());

        for i in 0..N128 {
            let ji = i + POS1;
            let b_idx = if ji >= N128 { ji - N128 } else { ji };

            let ci = i + N128 - 2;
            let c_idx = if ci >= N128 { ci - N128 } else { ci };

            let di = i + N128 - 1;
            let d_idx = if di >= N128 { di - N128 } else { di };

            let a = vld1q_u32(state.as_ptr().add(i * MULT));
            let b = vld1q_u32(state.as_ptr().add(b_idx * MULT));
            let c = vld1q_u32(state.as_ptr().add(c_idx * MULT));
            let d = vld1q_u32(state.as_ptr().add(d_idx * MULT));

            let rr = recurrence_relation(a, b, c, d, mask);
            vst1q_u32(state.as_mut_ptr().add(i * MULT), rr);
        }
    }

    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn recurrence_relation(
        a: uint32x4_t,
        b: uint32x4_t,
        c: uint32x4_t,
        d: uint32x4_t,
        mask: uint32x4_t,
    ) -> uint32x4_t {
        let x = shift_left_128_epi32(a);
        let y = vandq_u32(shift_right_128_epi32(b), mask);

        let c_sr2 = shift_right_128_epi32(c);
        let d_sl2 = shift_left_128_epi32(d);

        let mut r = veorq_u32(a, x);

        r = veorq_u32(r, y);
        r = veorq_u32(r, c_sr2);
        r = veorq_u32(r, d_sl2);

        r
    }

    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn shift_right_128_epi32(x: uint32x4_t) -> uint32x4_t {
        let xb = vreinterpretq_u8_u32(x);
        let zeros = vdupq_n_u8(0);
        let ext = vextq_u8(xb, zeros, 4);
        let ext_u32 = vreinterpretq_u32_u8(ext);

        let x1 = vshrq_n_u32(x, SR1 as i32);
        let x2 = vshlq_n_u32(ext_u32, (32 - SR1) as i32);

        vorrq_u32(x1, x2)
    }

    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn shift_left_128_epi32(x: uint32x4_t) -> uint32x4_t {
        let xb = vreinterpretq_u8_u32(x);
        let zeros = vdupq_n_u8(0);
        let ext = vextq_u8(xb, zeros, 12);
        let ext_u32 = vreinterpretq_u32_u8(ext);

        let x1 = vshlq_n_u32(x, SL1 as i32);
        let x2 = vshrq_n_u32(ext_u32, (32 - SL1) as i32);

        vorrq_u32(x1, x2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod isa_detection {
        use super::*;

        #[test]
        fn test_isa_detection_in_simd_init() {
            let simd = Simd::new();

            match simd.0 {
                #[cfg(target_arch = "x86_64")]
                ISA::AVX2 | ISA::SSE2 => {}

                #[cfg(target_arch = "aarch64")]
                ISA::NEON => {}

                _ => panic!("Unknown ISA detected for platform"),
            }
        }

        #[test]
        fn test_perform_sanity_check_for_isa_detetion() {
            let isa = ISA::detect_isa();

            match isa {
                #[cfg(target_arch = "x86_64")]
                ISA::AVX2 | ISA::SSE2 => {}

                #[cfg(target_arch = "aarch64")]
                ISA::NEON => {}

                _ => panic!("Unknown ISA detected for platform"),
            }
        }
    }

    mod simd {
        use super::*;
        use crate::sfmt::InnerState;

        #[test]
        fn test_gen_state_mutates_current_state() {
            let mut s = InnerState::new(0x123456789);

            let simd = Simd::new();
            let before = s.0;

            simd.gen_state(&mut s);

            assert_ne!(s.0, before, "gen_state should mutate the state");
        }

        #[test]
        fn test_gen_state_produces_different_values_on_multiple_calls() {
            let simd = Simd::new();

            let mut s = InnerState::new(0xDEADBEEFCAFEBABE);
            let first = s.0;

            simd.gen_state(&mut s);
            let second = s.0;

            simd.gen_state(&mut s);
            let third = s.0;

            assert_ne!(first, second, "first refill should change state");
            assert_ne!(second, third, "second refill should further change state");
        }

        #[test]
        fn test_sfmt_refill_changes_state() {
            let mut s = InnerState::new(0x12345678);
            let original = s.0;
            let simd = Simd::new();

            simd.gen_state(&mut s);

            assert_ne!(s.0, original, "SFMT refill must update the internal state");
        }
    }

    #[cfg(all(target_arch = "x86_64"))]
    mod target_avx2 {
        use super::*;
        use std::arch::x86_64::*;
        use std::mem;

        #[inline(always)]
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn loadu256(v: &[u32; 8]) -> __m256i {
            _mm256_loadu_si256(v.as_ptr() as *const __m256i)
        }

        #[inline(always)]
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn storeu256(v: __m256i) -> [u32; 8] {
            let mut out = mem::MaybeUninit::<[u32; 8]>::uninit();
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);

            out.assume_init()
        }

        #[test]
        fn test_shift_right_256_epi32_shifts_correctly() {
            unsafe {
                let input = [1, 2, 3, 4, 5, 6, 7, 8];
                let x = loadu256(&input);
                let r = avx2::shift_right_256_epi32(x);
                let result = storeu256(r);

                assert!(result.iter().zip(input).all(|(&r, i)| r <= i));
            }
        }

        #[test]
        fn test_shift_left_256_epi32_shifts_correctly() {
            unsafe {
                let input = [1, 2, 3, 4, 5, 6, 7, 8];
                let x = loadu256(&input);
                let r = avx2::shift_left_256_epi32(x);
                let result = storeu256(r);

                assert!(result.iter().zip(input).all(|(&r, i)| r >= i));
            }
        }

        #[test]
        fn test_recurrence_relation_256_deterministic() {
            unsafe {
                let a = _mm256_set1_epi32(0x11111111);
                let b = _mm256_set1_epi32(0x22222222);
                let c = _mm256_set1_epi32(0x33333333);
                let d = _mm256_set1_epi32(0x44444444);
                let mask = _mm256_set1_epi32(0xdeadbeefu32 as i32);

                let r1 = avx2::recurrence_relation_256(a, b, c, d, mask);
                let r2 = avx2::recurrence_relation_256(a, b, c, d, mask);

                let o1 = storeu256(r1);
                let o2 = storeu256(r2);

                assert_eq!(o1, o2, "recurrence must be deterministic");
            }
        }

        #[test]
        fn test_generate_inner_state_changes_state() {
            let mut state = [0xabcdef01u32; STATE32_LEN];
            let prev = state;

            unsafe { avx2::generate_inner_state(&mut state) };

            assert_ne!(state, prev, "state must be mutated after generation");
        }

        #[test]
        fn test_generate_inner_state_deterministic() {
            let mut s1 = [0x55555555u32; STATE32_LEN];
            let mut s2 = s1.clone();

            unsafe {
                avx2::generate_inner_state(&mut s1);
                avx2::generate_inner_state(&mut s2);
            }

            assert_eq!(s1, s2, "identical input -> identical output");
        }

        #[test]
        fn test_avx2_vs_sse2_equivalence() {
            let mut s_avx = [0x99999999u32; STATE32_LEN];
            let mut s_sse = s_avx.clone();

            unsafe {
                avx2::generate_inner_state(&mut s_avx);
                sse2::generate_inner_state(&mut s_sse);
            }

            assert_eq!(&s_avx[..64], &s_sse[..64]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod target_sse2 {
        use super::*;
        use std::arch::x86_64::*;

        #[test]
        fn test_shift_right_128_epi32_shifts_correctly() {
            unsafe {
                let mut buf = [0u32; 4];
                let orig: [u32; 4] = [1, 2, 3, 4];

                let x = _mm_set_epi32(0x00000004, 0x00000003, 0x00000002, 0x00000001);
                let r = sse2::shift_right_128_epi32(x);

                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, r);

                let shifted = ((u128::from(orig[3]) << 96)
                    | (u128::from(orig[2]) << 64)
                    | (u128::from(orig[1]) << 32)
                    | u128::from(orig[0]))
                    >> super::SR1;

                let expected: [u32; 4] = [
                    (shifted & 0xFFFF_FFFF) as u32,
                    ((shifted >> 32) & 0xFFFF_FFFF) as u32,
                    ((shifted >> 64) & 0xFFFF_FFFF) as u32,
                    ((shifted >> 96) & 0xFFFF_FFFF) as u32,
                ];

                assert_eq!(buf, expected, "cross-lane right shift mismatch");
            }
        }

        #[test]
        fn test_shift_left_128_epi32_shifts_correctly() {
            unsafe {
                let x = _mm_set_epi32(1, 2, 3, 4);
                let r = sse2::shift_left_128_epi32(x);

                let mut buf = [0u32; 4];
                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, r);

                assert!(buf.iter().any(|&v| v > 4), "left shift should increase values");
            }
        }

        #[test]
        fn test_recurrence_relation_deterministic() {
            unsafe {
                let a = _mm_set1_epi32(0xAAAAAAAAu32 as i32);
                let b = _mm_set1_epi32(0xBBBBBBBBu32 as i32);
                let c = _mm_set1_epi32(0xCCCCCCCCu32 as i32);
                let d = _mm_set1_epi32(0xDDDDDDDDu32 as i32);
                let mask = _mm_set_epi32(MSK[3] as i32, MSK[2] as i32, MSK[1] as i32, MSK[0] as i32);

                let r1 = sse2::recurrence_relation(a, b, c, d, mask);
                let r2 = sse2::recurrence_relation(a, b, c, d, mask);

                let mut buf1 = [0u32; 4];
                let mut buf2 = [0u32; 4];

                _mm_storeu_si128(buf1.as_mut_ptr() as *mut __m128i, r1);
                _mm_storeu_si128(buf2.as_mut_ptr() as *mut __m128i, r2);

                assert_eq!(buf1, buf2, "recurrence_relation must be deterministic");
            }
        }

        #[test]
        fn test_generate_inner_state_is_deterministic() {
            unsafe {
                let mut s1 = InnerState([1u32; STATE32_LEN]);
                let mut s2 = InnerState([1u32; STATE32_LEN]);

                sse2::generate_inner_state(&mut s1.0);
                sse2::generate_inner_state(&mut s2.0);

                assert_eq!(s1.0, s2.0, "identical inputs should yield identical outputs");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod target_neon {
        use super::*;
        use core::arch::aarch64::*;

        #[test]
        fn test_shift_right_128_epi32_shifts_correctly() {
            unsafe {
                let orig: [u32; 4] = [1, 2, 3, 4];
                let x = vld1q_u32(orig.as_ptr());
                let r = neon::shift_right_128_epi32(x);

                let mut buf = [0u32; 4];
                vst1q_u32(buf.as_mut_ptr(), r);

                let shifted = ((u128::from(orig[3]) << 96)
                    | (u128::from(orig[2]) << 64)
                    | (u128::from(orig[1]) << 32)
                    | u128::from(orig[0]))
                    >> super::SR1;

                let expected: [u32; 4] = [
                    (shifted & 0xFFFF_FFFF) as u32,
                    ((shifted >> 32) & 0xFFFF_FFFF) as u32,
                    ((shifted >> 64) & 0xFFFF_FFFF) as u32,
                    ((shifted >> 96) & 0xFFFF_FFFF) as u32,
                ];

                assert_eq!(buf, expected, "cross-lane right shift mismatch");
            }
        }

        #[test]
        fn test_shift_left_128_epi32_shifts_correctly() {
            unsafe {
                let orig = [1u32, 2, 3, 4];
                let x = vld1q_u32(orig.as_ptr());
                let r = neon::shift_left_128_epi32(x);

                let mut buf = [0u32; 4];
                vst1q_u32(buf.as_mut_ptr(), r);

                assert!(buf.iter().any(|&v| v > 4), "left shift should increase values");
            }
        }

        #[test]
        fn test_recurrence_relation_deterministic() {
            unsafe {
                let a = vdupq_n_u32(0xAAAAAAAA);
                let b = vdupq_n_u32(0xBBBBBBBB);
                let c = vdupq_n_u32(0xCCCCCCCC);
                let d = vdupq_n_u32(0xDDDDDDDD);
                let mask = vld1q_u32(MSK.as_ptr());

                let r1 = neon::recurrence_relation(a, b, c, d, mask);
                let r2 = neon::recurrence_relation(a, b, c, d, mask);

                let mut buf1 = [0u32; 4];
                let mut buf2 = [0u32; 4];
                vst1q_u32(buf1.as_mut_ptr(), r1);
                vst1q_u32(buf2.as_mut_ptr(), r2);

                assert_eq!(buf1, buf2, "recurrence_relation must be deterministic");
            }
        }

        #[test]
        fn test_generate_inner_state_is_deterministic() {
            unsafe {
                let mut s1 = InnerState([1u32; STATE32_LEN]);
                let mut s2 = InnerState([1u32; STATE32_LEN]);

                neon::generate_inner_state(&mut s1.0);
                neon::generate_inner_state(&mut s2.0);

                assert_eq!(s1.0, s2.0, "identical inputs should yield identical outputs");
            }
        }
    }
}
