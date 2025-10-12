pub(crate) mod sse2;

pub(crate) trait Engine<const N: usize> {
    type Lane;

    unsafe fn new(seed: u64) -> [Self::Lane; N];

    unsafe fn regenerate(state: &mut [Self::Lane; N]);
}
