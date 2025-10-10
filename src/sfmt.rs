use crate::simd::{PARITY, STATE32_LEN};

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

            s[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
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

pub(crate) struct Sfmt(pub(crate) InnerState);

impl Sfmt {
    #[inline(always)]
    pub(crate) fn new_seeded(seed: u64) -> Self {
        Self(InnerState::new(seed))
    }

    #[inline(always)]
    pub(crate) fn new() -> Self {
        let seed = unsafe { platform_seed() };
        Self(InnerState::new(seed))
    }
}

/// Generate a custom seed w/ help of underlying hardware.
#[inline(always)]
unsafe fn platform_seed() -> u64 {
    // NOTE: On x86_64, the `rdtsc` is reliable and generally available in all three OS.
    // It's fast as it avoids syscall overhead.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::asm;

        let mut lo: u32;
        let mut hi: u32;

        asm!("rdtsc", out("eax") lo, out("edx") hi);

        ((hi as u64) << 32) | (lo as u64)
    }

    // NOTE: On aarch64, we read the virtual counter `cntvct`. It provides high-res
    // monotonic value w/o syscall overhead.
    #[cfg(target_arch = "aarch64")]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
    unsafe {
        use std::arch::asm;

        let cnt: u64;
        asm!("mrs {0}, cntvct_el0", out(reg) cnt);

        cnt
    }

    // WARN: On Win-Aarch64, as usual, `cntvct` is not available (deemed as illegal instructions)
    // So using the `SysTime` is only viable option
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_os = "windows")]
    {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards");

        // Combine seconds and nanoseconds into a u64
        ((now.as_secs() as u64) << 32) ^ (now.subsec_nanos() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod sfmt {
        use super::*;

        #[test]
        fn test_sfmt_init_with_platform_seed_works() {
            let sfmt = Sfmt::new();

            assert!(
                sfmt.0 .0.iter().any(|&x| x != 0),
                "innerState slice must contain non-zero values"
            );
        }

        #[test]
        fn test_sfmt_init_with_custom_seed_works() {
            let sfmt = Sfmt::new_seeded(0x123456789);

            assert!(
                sfmt.0 .0.iter().any(|&x| x != 0),
                "innerState slice must contain non-zero values"
            );
        }
    }

    mod inner_state {
        use super::*;

        #[test]
        fn test_same_seeds_creates_same_states() {
            let s1 = InnerState::new(0x123456789);
            let s2 = InnerState::new(0x123456789);

            assert_eq!(s1.0, s2.0, "same seeds should create same states");
        }

        #[test]
        fn test_diff_seeds_creates_diff_states() {
            let s1 = InnerState::new(0x123456789);
            let s2 = InnerState::new(0x987654321);

            assert_ne!(s1.0, s2.0, "different seeds should create different states");
        }

        #[test]
        fn test_period_cert_has_odd_parity() {
            let mut s = InnerState::new(0xDEADBEEFCAFEBABE);

            let x = s.0[0] ^ s.0[1] ^ s.0[2] ^ s.0[3];
            let mut parity = 0u32;

            for i in 0..4 {
                parity ^= x & PARITY[i];
            }

            assert_eq!(parity.count_ones() % 2, 1, "period_cert must produce an odd parity");
        }
    }

    mod platform_seed_generation {
        use super::*;

        #[test]
        fn test_platform_seed_runtime_sanity() {
            let s1 = unsafe { platform_seed() };

            // NOTE: custom delay (for just in case)
            std::thread::sleep(std::time::Duration::from_millis(1));

            let s2 = unsafe { platform_seed() };

            assert_ne!(s1, s2, "platform_seed should produce different values");
            assert!(s1 != 0 && s2 != 0, "seed values must be non-zero");
        }
    }
}
