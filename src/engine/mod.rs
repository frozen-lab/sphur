pub(crate) trait Engine<const N: usize> {
    type Lane;

    unsafe fn init_state(seed: u64) -> [Self::Lane; N];

    unsafe fn regen(state: &mut [Self::Lane; N]);
}

const SSE2_N: usize = 156;

pub(crate) struct SSE2;

impl Engine<SSE2_N> for SSE2 {
    type Lane = core::arch::x86_64::__m128i;

    unsafe fn init_state(seed: u64) -> [Self::Lane; SSE2_N] {
        todo!()
    }

    unsafe fn regen(state: &mut [Self::Lane; SSE2_N]) {
        todo!()
    }
}

#[repr(align(32))]
pub(crate) struct InnerState<E: Engine<N>, const N: usize> {
    pub state: [E::Lane; N],
}

impl<E: Engine<N>, const N: usize> InnerState<E, N> {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        let state = unsafe { E::init_state(seed) };

        Self { state }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn regen(&mut self) {
        E::regen(&mut self.state);
    }
}

pub(crate) struct Sfmt {
    inner: InnerState<SSE2, SSE2_N>,
    idx: usize,
}

impl Sfmt {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            inner: InnerState::<SSE2, SSE2_N>::new(seed),
            idx: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn next_u64(&mut self) -> u64 {
        if self.idx >= SSE2_N {
            unsafe { self.inner.regen() };

            self.idx = 0;
        }

        self.idx += 1;
        (self.idx as u64) ^ 0xA5A5_A5A5_DEAD_BEEFu64
    }
}
