#[repr(align(32))]
pub(crate) struct State<E: crate::engine::Engine<N, N64, N32>, const N: usize, const N64: usize, const N32: usize> {
    state: [E::Lane; N],
    buf_u64: [u64; N64],
    buf_u32: [u32; N32],
    lane: usize,
    idx_u64: usize,
    idx_u32: usize,
}

impl<E: crate::engine::Engine<N, N64, N32>, const N: usize, const N64: usize, const N32: usize> State<E, N, N64, N32> {
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn new(seed: u64) -> Self {
        let state = E::new(seed);
        let buf_u64 = E::batch_u64(&state, 0);
        let buf_u32 = E::batch_u32(&state, 0);

        Self {
            state,
            buf_u64,
            buf_u32,
            lane: 1,
            idx_u64: 0,
            idx_u32: 0,
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn regen(&mut self) {
        E::regen(&mut self.state);
        self.reset_indices();
    }

    #[inline(always)]
    fn reset_indices(&mut self) {
        self.idx_u64 = 0;
        self.idx_u32 = 0;
        self.lane = 0;
    }

    // #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn gen_64(&mut self) -> u64 {
        // sanity check
        debug_assert!(self.idx_u64 <= N64);

        if self.idx_u64 == N64 {
            self.idx_u64 = 0;
            self.lane += 1;

            if self.lane >= N {
                self.regen();
            }

            // prefetch next lane
            self.buf_u64 = E::batch_u64(&self.state, self.lane);
        }

        let val = *self.buf_u64.get_unchecked(self.idx_u64);
        self.idx_u64 += 1;

        val
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn gen_32(&mut self) -> u32 {
        // sanity check
        debug_assert!(self.idx_u32 <= N32);

        if self.idx_u32 == N32 {
            self.idx_u32 = 0;
            self.lane += 1;

            if self.lane >= N {
                self.regen();
            }

            // prefetch next lane
            self.buf_u32 = E::batch_u32(&self.state, self.lane);
        }

        let val = *self.buf_u32.get_unchecked(self.idx_u32);
        self.idx_u32 += 1;

        val
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn batch_u64(&mut self) -> [u64; N64] {
        if self.lane >= N {
            self.regen();
        }

        self.lane += 1;
        E::batch_u64(&self.state, self.lane)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn batch_u32(&mut self) -> [u32; N32] {
        if self.lane >= N {
            self.regen();
        }

        self.lane += 1;
        E::batch_u32(&self.state, self.lane)
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    mod sse_engine {
        use super::*;
        use crate::engine::sse::{SSE, SSE_STATE_LEN};

        #[test]
        fn test_sanity_check_basic_u64_generation_in_state() {
            unsafe {
                let mut st = State::<SSE, SSE_STATE_LEN, 2, 4>::new(1234);

                let a = st.gen_64();
                let b = st.gen_64();

                assert_ne!(a, b, "Consecutive u64 values should differ");

                // sanity check
                for _ in 0..(SSE_STATE_LEN * 4) {
                    let _ = st.gen_64();
                }
            }
        }

        #[test]
        fn test_sanity_check_basic_u32_generation_in_state() {
            unsafe {
                let mut st = State::<SSE, SSE_STATE_LEN, 2, 4>::new(5678);

                let a = st.gen_32();
                let b = st.gen_32();

                assert_ne!(a, b, "Consecutive u32 values should differ");

                // sanity check
                for _ in 0..(SSE_STATE_LEN * 8) {
                    let _ = st.gen_32();
                }
            }
        }

        #[test]
        fn test_batch_u64_and_u32_generation() {
            unsafe {
                let mut st = State::<SSE, SSE_STATE_LEN, 2, 4>::new(42);

                let b64 = st.batch_u64();
                let b32 = st.batch_u32();

                assert_eq!(b64.len(), 2);
                assert_eq!(b32.len(), 4);

                // make sure not all zero
                assert!(b64.iter().any(|&x| x != 0));
                assert!(b32.iter().any(|&x| x != 0));
            }
        }

        #[test]
        fn test_determinism_same_seed() {
            unsafe {
                let mut a = State::<SSE, SSE_STATE_LEN, 2, 4>::new(9999);
                let mut b = State::<SSE, SSE_STATE_LEN, 2, 4>::new(9999);

                for _ in 0..1000 {
                    assert_eq!(a.gen_32(), b.gen_32(), "States w/ same seed must be deterministic");
                }
            }
        }

        #[test]
        fn test_alignment_safety() {
            use std::mem::{align_of, size_of};

            assert_eq!(align_of::<State<SSE, SSE_STATE_LEN, 2, 4>>(), 32);
            assert_eq!(size_of::<std::arch::x86_64::__m128i>(), 16);
        }
    }
}
