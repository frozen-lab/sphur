fn main() {
    let mut rng = sphur::Sphur::new_seeded(42);
    let mut sum = 0u64;

    for _ in 0..10_000_000 {
        sum = sum.wrapping_add(rng.next_u64());
    }

    // prevents optimizations !?
    println!("{}", sum);
}
