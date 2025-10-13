#[cfg(target_arch = "x86_64")]
pub(crate) mod sse;

const PARITY: [u32; 4] = [0x00000001, 0x00000000, 0x00000000, 0x13c9e684];

pub(crate) trait Engine<const N: usize, const N64: usize, const N32: usize> {
    type Lane;

    unsafe fn new(seed: u64) -> [Self::Lane; N];
    unsafe fn regen(state: &mut [Self::Lane; N]);

    unsafe fn next_u64(state: &[Self::Lane; N], lane: usize, idx: usize) -> u64;
    unsafe fn next_u32(state: &[Self::Lane; N], lane: usize, idx: usize) -> u32;

    unsafe fn batch_u64(state: &[Self::Lane; N], lane: usize) -> [u64; N64];
    unsafe fn batch_u32(state: &[Self::Lane; N], lane: usize) -> [u32; N32];
}

#[inline(always)]
pub(super) fn init_engine_state<Lane, const N: usize, const TOTAL_WORDS: usize>(
    seed: u64,
    pack: impl Fn([u32; 4]) -> Lane,
) -> [Lane; N] {
    let mut s = [0u32; TOTAL_WORDS];

    s[0] = seed as u32;
    s[1] = (seed >> 32) as u32;

    for i in 2..TOTAL_WORDS {
        let prev = s[i - 1];
        s[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
    }

    period_certify(&mut s);

    core::array::from_fn(|i| {
        let base = i * 4;
        pack([s[base], s[base + 1], s[base + 2], s[base + 3]])
    })
}

#[inline(always)]
fn period_certify<const TOTAL_WORDS: usize>(state: &mut [u32; TOTAL_WORDS]) {
    let mut check = 0u32;
    let inner = state[0] ^ state[1] ^ state[2] ^ state[3];

    for i in 0..4 {
        check ^= inner & PARITY[i];
    }

    if check.count_ones() & 1 == 0 {
        'outer: for i in 0..4 {
            for bit in 0..32 {
                if (PARITY[i] >> bit) & 1 != 0 {
                    state[i] ^= 1u32 << bit;

                    break 'outer;
                }
            }
        }
    }
}
