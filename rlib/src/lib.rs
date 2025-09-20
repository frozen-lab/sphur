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

pub struct Sphur {
    inner: sphur_t,
}

impl Sphur {
    pub fn new() -> Option<Self> {
        let mut state = sphur_t {
            _seeds: [0; 8],
            _rands: [0; 4],
            _rcnt: 0,
            _rpos: 0,
            _simd_ext: SphurSimdExt::SPHUR_SIMD_SSE2,
        };

        let res = unsafe { sphur_init_wrapper(&mut state) };
        if res == 0 {
            Some(Self { inner: state })
        } else {
            None
        }
    }

    pub fn new_seeded(seed: u64) -> Option<Self> {
        let mut state = sphur_t {
            _seeds: [0; 8],
            _rands: [0; 4],
            _rcnt: 0,
            _rpos: 0,
            _simd_ext: SphurSimdExt::SPHUR_SIMD_SSE2,
        };

        let res = unsafe { sphur_init_seeded_wrapper(&mut state, seed) };

        if res == 0 {
            Some(Self { inner: state })
        } else {
            None
        }
    }

    pub fn rand(&mut self) -> u64 {
        unsafe { sphur_gen_rand_wrapper(&mut self.inner) }
    }

    pub fn rand_bool(&mut self) -> bool {
        unsafe { sphur_gen_bool_wrapper(&mut self.inner) != 0 }
    }

    pub fn rand_range(&mut self, min: u64, max: u64) -> u64 {
        unsafe { sphur_gen_rand_range_wrapper(&mut self.inner, min, max) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_default() {
        let rng = Sphur::new();
        assert!(rng.is_some(), "Failed to initialize default Sphur RNG");
    }

    #[test]
    fn test_init_seeded() {
        let rng = Sphur::new_seeded(123456789);
        assert!(rng.is_some(), "Failed to initialize seeded Sphur RNG");
    }

    #[test]
    fn test_rand_changes() {
        let mut rng = Sphur::new().expect("Failed to init RNG");
        let r1 = rng.rand();
        let r2 = rng.rand();
        assert_ne!(
            r1, r2,
            "Two consecutive random numbers should not always be equal"
        );
    }

    #[test]
    fn test_rand_bool_valid() {
        let mut rng = Sphur::new().expect("Failed to init RNG");
        for _ in 0..100 {
            let b = rng.rand_bool();
            assert!(b == true || b == false, "rand_bool returned invalid value");
        }
    }

    #[test]
    fn test_rand_range_bounds() {
        let mut rng = Sphur::new().expect("Failed to init RNG");
        for _ in 0..100 {
            let val = rng.rand_range(1, 6);
            assert!(
                (1..=6).contains(&val),
                "rand_range returned value out of bounds: {}",
                val
            );
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = Sphur::new_seeded(42).expect("Failed to init RNG");
        let mut rng2 = Sphur::new_seeded(1337).expect("Failed to init RNG");

        let v1: Vec<u64> = (0..5).map(|_| rng1.rand()).collect();
        let v2: Vec<u64> = (0..5).map(|_| rng2.rand()).collect();

        assert_ne!(
            v1, v2,
            "Different seeds should produce different random sequences"
        );
    }
}
