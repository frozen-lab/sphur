const MEXP: usize = 19937;
const N: usize = 156;
const STATE32_LEN: usize = N * 4;
const PARITY: [u32; 4] = [0x00000001, 0x00000000, 0x00000000, 0x13c9e684];

#[repr(align(32))]
pub(crate) struct InnerState(pub(crate) [u32; STATE32_LEN]);

impl InnerState {
    pub(crate) fn new(seed: u64) -> Self {
        let mut s = [0u32; STATE32_LEN];

        s[0] = seed as u32;
        s[1] = (seed >> 32) as u32;

        // mersenne twister style state expansion w/ `s[i] = f(s[i-1], i)`
        for i in 2..STATE32_LEN {
            let prev = s[i - 1];

            s[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }

        // NOTE: period certification to ensure full period
        Self::period_cert(&mut s);

        Self(s)
    }

    /// Ensures that the initialized SFMT state satisfies the **period certification**
    /// condition required for maximal period (`2^19937 - 1`).
    fn period_cert(state: &mut [u32; STATE32_LEN]) {
        let inner = state[0] ^ state[1] ^ state[2] ^ state[3];

        // parity check
        let mut check: u32 = 0;

        for i in 0..4 {
            check ^= inner & PARITY[i];
        }

        let ones = check.count_ones();

        if (ones & 1) == 0 {
            // NOTE: As bit parity is even, we flip the lowst order bit found in parity

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
}
