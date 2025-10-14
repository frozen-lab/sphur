#[cfg(target_arch = "x86_64")]
pub(crate) mod sse;

const PARITY: [u32; 4] = [0x00000001, 0x00000000, 0x00000000, 0x13c9e684];

pub(crate) trait Engine<const N: usize, const N64: usize, const N32: usize> {
    type Lane;

    unsafe fn new(seed: u64) -> [Self::Lane; N];
    unsafe fn regen(state: &mut [Self::Lane; N]);

    unsafe fn batch_u64(state: &[Self::Lane; N], lane: usize) -> [u64; N64];
    unsafe fn batch_u32(state: &[Self::Lane; N], lane: usize) -> [u32; N32];
}

#[inline(always)]
pub(super) fn init_engine_state<Lane, const N: usize, const TOTAL_WORDS: usize>(
    seed: u64,
    pack: impl Fn([u32; 4]) -> Lane,
) -> [Lane; N] {
    // NOTE: Sanity check cause the manual loop unroll assumes this condition
    debug_assert!(TOTAL_WORDS % 4 == 0, "TOTAL_WORDS must be multiple of 4");

    let mut s = [0u32; TOTAL_WORDS];

    s[0] = seed as u32;
    s[1] = (seed >> 32) as u32;

    const MULT: u32 = 1812433253u32;

    //
    // NOTE: Manual 4 way unroll for throughput
    //

    let mut i = 2usize;

    while i + 3 < TOTAL_WORDS {
        let prev0 = s[i - 1];
        let t0 = prev0 ^ (prev0 >> 30);
        s[i] = t0.wrapping_mul(MULT).wrapping_add(i as u32);

        let prev1 = s[i];
        let t1 = prev1 ^ (prev1 >> 30);
        s[i + 1] = t1.wrapping_mul(MULT).wrapping_add((i + 1) as u32);

        let prev2 = s[i + 1];
        let t2 = prev2 ^ (prev2 >> 30);
        s[i + 2] = t2.wrapping_mul(MULT).wrapping_add((i + 2) as u32);

        let prev3 = s[i + 2];
        let t3 = prev3 ^ (prev3 >> 30);
        s[i + 3] = t3.wrapping_mul(MULT).wrapping_add((i + 3) as u32);

        i += 4;
    }

    period_certify(&mut s);

    core::array::from_fn(|idx| {
        let base = idx * 4;
        pack([s[base], s[base + 1], s[base + 2], s[base + 3]])
    })
}

#[inline(always)]
fn period_certify<const TOTAL_WORDS: usize>(state: &mut [u32; TOTAL_WORDS]) {
    let inner = state[0] ^ state[1] ^ state[2] ^ state[3];
    let mut check = (inner & PARITY[0]) ^ (inner & PARITY[1]) ^ (inner & PARITY[2]) ^ (inner & PARITY[3]);

    if check.count_ones() & 1 == 0 {
        for i in 0..4 {
            let p = PARITY[i];

            if p != 0 {
                let bit = p.trailing_zeros();
                state[i] ^= 1u32 << bit;

                break;
            }
        }
    }
}
