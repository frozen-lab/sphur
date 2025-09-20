#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub enum SphurSimdExt {
    SPHUR_SIMD_AVX2 = 0,
    SPHUR_SIMD_SSE2 = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub struct sphur_t {
    pub _seeds: [u64; 8],
    pub _rands: [u64; 4],
    pub _rcnt: u32,
    pub _rpos: u32,
    pub _simd_ext: SphurSimdExt,
}

#[link(name = "sphur_wrapper")]
unsafe extern "C" {
    pub fn sphur_init_wrapper(state: *mut sphur_t) -> i32;
    pub fn sphur_init_seeded_wrapper(state: *mut sphur_t, seed: u64) -> i32;
    pub fn sphur_gen_rand_wrapper(state: *mut sphur_t) -> u64;
    pub fn sphur_gen_bool_wrapper(state: *mut sphur_t) -> i32;
    pub fn sphur_gen_rand_range_wrapper(state: *mut sphur_t, min: u64, max: u64) -> u64;
}

/// Sphūr is SIMD accelerated Pseudo-Random Number Generator
///
/// Internally, it keeps a small buffer of random numbers, and
/// it uses multiple "sub-seeds" for better distribution.
pub struct Sphur(sphur_t);

impl Default for Sphur {
    fn default() -> Self {
        Self(sphur_t {
            _seeds: [0; 8],
            _rands: [0; 4],
            _rcnt: 0,
            _rpos: 0,
            _simd_ext: SphurSimdExt::SPHUR_SIMD_SSE2,
        })
    }
}

/// A custom result type
pub type SphurResult<T> = Result<T, SphurError>;

/// Errors returned by Sphur PRNG
#[derive(Debug)]
pub enum SphurError {
    /// Represents initilization error
    InitError,

    /// Internal RNG failure (SIMD/gen failure)
    InternalError,

    /// Invalid arguments
    InvalidRange,
}

impl Sphur {
    /// Initialize sphur state
    ///
    /// ## Returns
    ///
    /// - `Ok(Sphur)` on success  
    /// - `Err(SphurError::InitError)` on failure
    ///
    /// ## Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new().expect("failed to init RNG");
    /// let val = rng.rand().expect("failed to generate rand");
    /// assert!(val <= u64::MAX);
    /// ```
    pub fn new() -> SphurResult<Self> {
        let mut state = Sphur::default();
        let res = unsafe { sphur_init_wrapper(&mut state.0) };

        if res == 0 {
            Ok(state)
        } else {
            Err(SphurError::InitError)
        }
    }

    /// Initialize sphur state with an initial seed
    ///
    /// ## Returns
    ///
    /// - `Ok(Sphur)` on success  
    /// - `Err(SphurError::InitError)` on failure
    ///
    /// ## Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new_seeded(12345).expect("failed to init RNG with seed");
    /// let val = rng.rand().expect("failed to generate rand");
    /// assert!(val <= u64::MAX);
    /// ```
    pub fn new_seeded(seed: u64) -> SphurResult<Self> {
        let mut state = Sphur::default();
        let res = unsafe { sphur_init_seeded_wrapper(&mut state.0, seed) };

        if res == 0 {
            Ok(state)
        } else {
            Err(SphurError::InitError)
        }
    }

    /// Generate a 64-bit random number
    ///
    /// NOTE: It uses a small buffer of 2/4 numbers, based on the SIMD
    /// capabilities available.
    ///
    /// ## Returns
    ///
    /// - `Ok(u64)` random number  
    /// - `Err(SphurError::InternalError)` on failure
    ///
    /// ## Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new().expect("failed to init RNG");
    /// let val = rng.rand().expect("failed to generate rand");
    /// assert!(val <= u64::MAX);
    /// ```
    pub fn rand(&mut self) -> SphurResult<u64> {
        let val = unsafe { sphur_gen_rand_wrapper(&mut self.0) };

        if val == u64::MAX {
            Err(SphurError::InternalError)
        } else {
            Ok(val)
        }
    }

    /// Generate a random boolean (`true` or `false`)
    ///
    /// ## Returns
    ///
    /// - `Ok(bool)` on success  
    /// - `Err(SphurError::InternalError)` on failure
    ///
    /// ## Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new().expect("failed to init RNG");
    /// let flag = rng.rand_bool().expect("failed to generate bool");
    /// assert!(flag == true || flag == false);
    /// ```
    pub fn rand_bool(&mut self) -> SphurResult<bool> {
        let val = unsafe { sphur_gen_bool_wrapper(&mut self.0) };

        if val == -1 {
            Err(SphurError::InternalError)
        } else {
            Ok(val != 0)
        }
    }

    /// Generate a random number in a range `[min, max]` (inclusive)
    ///
    /// NOTE: Rejection sampling is used to avoid modulo bias.
    ///
    /// ## Returns
    ///
    /// - `Ok(u64)` on success  
    /// - `Err(SphurError::InvalidRange)` if `min > max`  
    /// - `Err(SphurError::InternalError)` on failure
    ///
    /// ## Example
    ///
    /// ```
    /// use sphur::Sphur;
    ///
    /// let mut rng = Sphur::new().expect("failed to init RNG");
    /// let val = rng.rand_range(10, 20).expect("failed to generate rand in range");
    /// assert!(val >= 10 && val <= 20);
    /// ```
    pub fn rand_range(&mut self, min: u64, max: u64) -> SphurResult<u64> {
        let val = unsafe { sphur_gen_rand_range_wrapper(&mut self.0, min, max) };

        if min > max {
            return Err(SphurError::InvalidRange);
        }

        if val == u64::MAX {
            return Err(SphurError::InternalError);
        }

        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_default() {
        let rng = Sphur::new();

        assert!(rng.is_ok(), "failed to init RNG with default seed");
    }

    #[test]
    fn test_init_seeded() {
        let rng = Sphur::new_seeded(42);

        assert!(rng.is_ok(), "failed to init RNG with custom seed");
    }

    #[test]
    fn test_rand_changes() {
        let mut rng = Sphur::new().expect("failed to init RNG");

        let a = rng.rand().expect("failed to generate first rand");
        let b = rng.rand().expect("failed to generate second rand");

        assert_ne!(
            a, b,
            "two successive rand calls should differ (with high probability)"
        );
    }

    #[test]
    fn test_rand_bool_valid() {
        let mut rng = Sphur::new().expect("failed to init RNG");
        let val = rng.rand_bool().expect("failed to generate bool");

        assert!(
            val == true || val == false,
            "rand_bool must produce a valid boolean"
        );
    }

    #[test]
    fn test_rand_range_bounds() {
        let mut rng = Sphur::new().expect("failed to init RNG");

        for _ in 0..100 {
            let val = rng
                .rand_range(5, 15)
                .expect("failed to generate rand in range");

            assert!(
                val >= 5 && val <= 15,
                "rand_range produced out-of-bounds value: {val}"
            );
        }
    }

    #[test]
    fn test_invalid_range() {
        let mut rng = Sphur::new().expect("failed to init RNG");
        let result = rng.rand_range(20, 10);

        assert!(matches!(result, Err(SphurError::InvalidRange)));
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = Sphur::new_seeded(111).expect("failed to init RNG with seed 111");
        let mut rng2 = Sphur::new_seeded(222).expect("failed to init RNG with seed 222");

        let v1 = rng1.rand().expect("failed to generate rand from rng1");
        let v2 = rng2.rand().expect("failed to generate rand from rng2");

        assert_ne!(
            v1, v2,
            "different seeds should produce different first outputs (with high probability)"
        );
    }
}
