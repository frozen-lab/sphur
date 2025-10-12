use core::arch::x86_64::*;

pub(crate) const SSE2_STATE_LEN: usize = 156;
pub(crate) struct SSE2;

impl super::Engine<SSE2_STATE_LEN> for SSE2 {
    type Lane = __m128i;

    unsafe fn new(seed: u64) -> [Self::Lane; SSE2_STATE_LEN] {
        todo!()
    }

    unsafe fn regenerate(state: &mut [Self::Lane; SSE2_STATE_LEN]) {
        todo!()
    }
}
