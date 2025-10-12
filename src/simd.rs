use crate::{
    engine::sse2::{SSE2, SSE2_STATE_LEN},
    state::State,
};

pub(crate) enum SIMD {
    Sse2(State<SSE2, SSE2_STATE_LEN>),
}

impl SIMD {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        match ISA::detect_isa() {
            ISA::SSE2 => Self::Sse2(State::<SSE2, SSE2_STATE_LEN>::new(seed)),
            ISA::NEON => todo!("NEON engine init"),
        }
    }

    #[inline(always)]
    pub(crate) fn block_size(&self) -> usize {
        match self {
            Self::Sse2(_) => SSE2_STATE_LEN,
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn regenerate(&mut self) {
        match self {
            Self::Sse2(inner) => inner.regenerate(),
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(unused)]
enum ISA {
    // SSE2 is used as default for all x86_64 CPU's
    SSE2,

    // Neon is used as default for all aarch64 CPU's
    NEON,
}

impl ISA {
    #[inline(always)]
    fn detect_isa() -> ISA {
        #[cfg(target_arch = "x86_64")]
        {
            ISA::SSE2
        }

        #[cfg(target_arch = "aarch64")]
        {
            ISA::NEON
        }
    }
}
