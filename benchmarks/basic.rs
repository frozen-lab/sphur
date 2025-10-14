use sphur::Sphur;
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 10_000_000;

fn main() {
    let mut rng = Sphur::new_seeded(0xDEADBEEF_u64);

    // warmup
    for _ in 0..1_000 {
        black_box(rng.next_u64());
    }

    let mut _acc: u64 = 0;
    let start = Instant::now();

    for _ in 0..ITERS {
        // NOTE: We consume result to avoid optimizations
        _acc = _acc.wrapping_add(black_box(rng.next_u64()));
    }

    let elapsed = start.elapsed();
    let thpt = elapsed.as_nanos().wrapping_div(ITERS as u128);

    println!("iters={}, thpt={}, time_ns={}", ITERS, thpt, elapsed.as_nanos());
}
