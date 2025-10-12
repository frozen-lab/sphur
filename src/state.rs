#[repr(align(32))]
pub(crate) struct State<E: crate::engine::Engine<N>, const N: usize> {
    pub state: [E::Lane; N],
}

impl<E: crate::engine::Engine<N>, const N: usize> State<E, N> {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: unsafe { E::new(seed) },
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn regenerate(&mut self) {
        E::regenerate(&mut self.state);
    }
}
