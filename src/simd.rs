#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(unused)]
enum ISA {
    // This ISA is an upgrade over SSE2 if available at runtime
    AVX2,

    // This ISA is default on x64 (x86_64), as it's virtually available on all x86_64 CPU's
    SSE2,

    // This ISA is default for aarch64, as its vurtually available on all aarch64 CPU's
    NEON,
}

impl ISA {
    fn detect_isa() -> ISA {
        // NOTE: On x86_64 we upgrade to AVX2 if available, otherwise
        // treat SSE2 as baseline
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return ISA::AVX2;
            }

            ISA::SSE2
        }

        #[cfg(target_arch = "aarch64")]
        return ISA::NEON;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod isa_detection {
        use super::*;

        #[test]
        fn test_perform_sanity_check_for_isa_detetion() {
            let isa = ISA::detect_isa();

            #[cfg(target_arch = "x86_64")]
            match isa {
                ISA::AVX2 | ISA::SSE2 => {}
                _ => panic!("Unknown ISA detected for x86_64"),
            }

            #[cfg(target_arch = "aarch64")]
            match isa {
                ISA::NEON => {}
                _ => panic!("Unknown ISA detected for aarch64"),
            }
        }
    }
}
