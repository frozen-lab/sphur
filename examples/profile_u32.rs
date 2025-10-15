use sphur::Sphur;
use std::hint::black_box;

const ITERS: usize = 100_000_000;

fn main() {
    let mut rng = Sphur::new_seeded(0xDEADBEEF);
    for _ in 0..1_000 {
        black_box(rng.next_u32());
    }

    for _ in 0..ITERS {
        black_box(rng.next_u32());
    }
}
