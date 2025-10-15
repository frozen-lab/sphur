#[repr(align(16))]
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

    #[inline(always)]
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
