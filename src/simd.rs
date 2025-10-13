use crate::state::State;

#[cfg(target_arch = "x86_64")]
use crate::engine::sse::{SSE, SSE_STATE_LEN};

pub(crate) enum SIMD {
    #[cfg(target_arch = "x86_64")]
    Sse(State<SSE, SSE_STATE_LEN>),
}

impl SIMD {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        match ISA::detect_isa() {
            #[cfg(target_arch = "x86_64")]
            ISA::SSE => Self::Sse(State::<SSE, SSE_STATE_LEN>::new(seed)),

            #[cfg(target_arch = "aarch64")]
            ISA::NEON => todo!(),
        }
    }

    #[inline(always)]
    pub(crate) fn block_size(&self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(_) => SSE_STATE_LEN,

            _ => todo!(),
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn regenerate(&mut self) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(inner) => inner.regenerate(),

            _ => todo!(),
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(unused)]
enum ISA {
    // SSE is used as default for all x86_64 CPU's
    //
    // NOTE: We upgrade to SSSE3, SSE4.1, etc. as per availabilty at runtime
    #[cfg(target_arch = "x86_64")]
    SSE,

    // Neon is used as default for all aarch64 CPU's
    #[cfg(target_arch = "aarch64")]
    NEON,
}

impl ISA {
    #[inline(always)]
    fn detect_isa() -> ISA {
        #[cfg(target_arch = "x86_64")]
        {
            ISA::SSE
        }

        #[cfg(target_arch = "aarch64")]
        {
            ISA::NEON
        }
    }
}
