use crate::state::State;

#[cfg(target_arch = "x86_64")]
use crate::engine::sse::{SSE, SSE_N32, SSE_N64, SSE_STATE_LEN};

#[cfg(target_arch = "aarch64")]
use crate::engine::neon::{NEON, NEON_N32, NEON_N64, NEON_STATE_LEN};

pub(crate) enum SIMD {
    #[cfg(target_arch = "x86_64")]
    Sse(State<SSE, SSE_STATE_LEN, SSE_N64, SSE_N32>),

    #[cfg(target_arch = "aarch64")]
    Neon(State<NEON, NEON_STATE_LEN, NEON_N64, NEON_N32>),
}

impl SIMD {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        match ISA::detect_isa() {
            #[cfg(target_arch = "x86_64")]
            ISA::SSE => unsafe { Self::Sse(State::<SSE, SSE_STATE_LEN, SSE_N64, SSE_N32>::new(seed)) },

            #[cfg(target_arch = "aarch64")]
            ISA::NEON => unsafe { Self::Neon(State::<NEON, NEON_STATE_LEN, NEON_N64, NEON_N32>::new(seed)) },
        }
    }

    #[inline(always)]
    pub(crate) fn next_u64(&mut self) -> u64 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(state) => unsafe { state.gen_64() },

            #[cfg(target_arch = "aarch64")]
            Self::Neon(state) => unsafe { state.gen_64() },
        }
    }

    #[inline(always)]
    pub(crate) fn next_u32(&mut self) -> u32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(state) => unsafe { state.gen_32() },

            #[cfg(target_arch = "aarch64")]
            Self::Neon(state) => unsafe { state.gen_32() },
        }
    }

    #[inline(always)]
    pub(crate) fn batch_u64(&mut self, buf: &mut [u64]) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(_) => unsafe {
                self.fill_u64_buf(buf, buf.len());
            },

            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => unsafe {
                self.fill_u64_buf(buf, buf.len());
            },
        }
    }

    #[inline(always)]
    pub(crate) fn batch_u32(&mut self, buf: &mut [u32]) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(_) => unsafe {
                self.fill_u32_buf(buf, buf.len());
            },

            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => unsafe {
                self.fill_u32_buf(buf, buf.len());
            },
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn fill_u64_buf(&mut self, buf: &mut [u64], len: usize) {
        // edge case
        if len == 0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        const BATCH: usize = SSE_N64;

        #[cfg(target_arch = "aarch64")]
        const BATCH: usize = NEON_N64;

        const UNROLL: usize = 4;
        const CHUNK: usize = BATCH * UNROLL;

        let state_ref = match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(s) => s,

            #[cfg(target_arch = "aarch64")]
            Self::Neon(n) => n,
        };

        let dst = buf.as_mut_ptr();

        // edge case
        if len < BATCH {
            let b = state_ref.batch_u64();
            core::ptr::copy_nonoverlapping(b.as_ptr(), dst, len);

            return;
        }

        let main_iters = len / CHUNK;

        let mut i = 0usize;
        let mut iters = main_iters;

        while iters != 0 {
            let b0 = state_ref.batch_u64();
            let b1 = state_ref.batch_u64();
            let b2 = state_ref.batch_u64();
            let b3 = state_ref.batch_u64();

            core::ptr::copy_nonoverlapping(b0.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            core::ptr::copy_nonoverlapping(b1.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            core::ptr::copy_nonoverlapping(b2.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            core::ptr::copy_nonoverlapping(b3.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            iters -= 1;
        }

        //
        // slow path (iif buf_len is odd)
        //

        let remaining = len - (main_iters * CHUNK);
        let mut rem_written = 0usize;

        while rem_written + BATCH <= remaining {
            let b = state_ref.batch_u64();
            core::ptr::copy_nonoverlapping(b.as_ptr(), dst.add(i + rem_written), BATCH);

            rem_written += BATCH;
        }

        if rem_written < remaining {
            let b = state_ref.batch_u64();
            core::ptr::write(dst.add(i + rem_written), b[0]);
        }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn fill_u32_buf(&mut self, buf: &mut [u32], len: usize) {
        // edge case
        if len == 0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        const BATCH: usize = SSE_N32;

        #[cfg(target_arch = "aarch64")]
        const BATCH: usize = NEON_N32;

        const UNROLL: usize = 4;
        const CHUNK: usize = BATCH * UNROLL;

        let state_ref = match self {
            #[cfg(target_arch = "x86_64")]
            Self::Sse(s) => s,

            #[cfg(target_arch = "aarch64")]
            Self::Neon(n) => n,
        };

        let dst = buf.as_mut_ptr();

        // edge case
        if len < BATCH {
            let b = state_ref.batch_u32();
            core::ptr::copy_nonoverlapping(b.as_ptr(), dst, len);

            return;
        }

        let main_iters = len / CHUNK;

        let mut i = 0usize;
        let mut iters = main_iters;

        while iters != 0 {
            let b0 = state_ref.batch_u32();
            let b1 = state_ref.batch_u32();
            let b2 = state_ref.batch_u32();
            let b3 = state_ref.batch_u32();

            core::ptr::copy_nonoverlapping(b0.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            core::ptr::copy_nonoverlapping(b1.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            core::ptr::copy_nonoverlapping(b2.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            core::ptr::copy_nonoverlapping(b3.as_ptr(), dst.add(i), BATCH);
            i += BATCH;

            iters -= 1;
        }

        //
        // slow path (iif buf_len is odd)
        //

        let remaining = len - (main_iters * CHUNK);
        let mut rem_written = 0usize;

        while rem_written + BATCH <= remaining {
            let b = state_ref.batch_u32();
            core::ptr::copy_nonoverlapping(b.as_ptr(), dst.add(i + rem_written), BATCH);

            rem_written += BATCH;
        }

        if rem_written < remaining {
            let b = state_ref.batch_u32();
            core::ptr::write(dst.add(i + rem_written), b[0]);
        }
    }
}

/// Generate a custom seed w/ help from available hardware
#[inline(always)]
pub(crate) fn platform_seed() -> u64 {
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

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
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
