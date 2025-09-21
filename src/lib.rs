#![allow(unused)]

/// Sphūr is SIMD accelerated Pseudo-Random Number Generator.
pub struct Sphur {
    /// Isa (Instruction Set Architecture) available at runtime
    isa: ISA,

    /// Current idx of the state being used for twistter
    idx: usize,
}

impl Sphur {
    pub fn new() -> SphurResult<Self> {
        Ok(Self {
            isa: detect_isa(),
            idx: 0,
        })
    }
}

/// Custom result type for [Sphur]
pub type SphurResult<T> = Result<T, SphurError>;

/// Types of Error exposed by [Sphur]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SphurError {
    InitError,
}

enum ISA {
    // This SIMD ISA is an upgrade over SSE2 if available at runtime
    AVX2,

    // This SIMD ISA is default on x64 (x86_64), as it's virtually
    // available on all x64 CPU's
    SSE2,

    // Neon is vurtually available on all aarch64 CPU's
    NEON,
}

#[cfg(target_arch = "x86_64")]
fn detect_isa() -> ISA {
    if is_x86_feature_detected!("avx2") {
        return ISA::AVX2;
    }

    ISA::SSE2
}

#[cfg(target_arch = "aarch64")]
fn detect_isa() -> ISA {
    ISA::NEON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_isa_returns_valid_value() {
        let isa = detect_isa();
        match isa {
            ISA::AVX2 | ISA::SSE2 | ISA::NEON => {}
            _ => panic!("Unknown ISA detected"),
        }
    }
}
