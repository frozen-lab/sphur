#[repr(align(32))]
pub(crate) struct State<E: crate::engine::Engine<N>, const N: usize> {
    state: [E::Lane; N],
    lane: usize,
    idx_u64: usize,
    idx_u32: usize,
    idx_u16: usize,
    idx_u8: usize,
}

impl<E: crate::engine::Engine<N>, const N: usize> State<E, N> {
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn new(seed: u64) -> Self {
        Self {
            state: E::new(seed),
            lane: 0,
            idx_u64: 0,
            idx_u32: 0,
            idx_u16: 0,
            idx_u8: 0,
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn regenerate(&mut self) {
        E::regen(&mut self.state);
        self.reinit_all_idx_and_lane();
    }

    #[inline(always)]
    fn reinit_all_idx_and_lane(&mut self) {
        self.idx_u64 = 0;
        self.idx_u32 = 0;
        self.idx_u16 = 0;
        self.idx_u8 = 0;

        self.lane = 0;
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn gen_64(&mut self) -> u64 {
        if self.idx_u64 >= 2 || self.lane >= N {
            self.regenerate();
        }

        self.idx_u64 += 1;

        E::gen_u64(&self.state, self.lane, self.idx_u64)
    }
}
