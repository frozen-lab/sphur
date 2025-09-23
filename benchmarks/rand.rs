use sphur::Sphur;
use std::fs::File;
use std::io::Write;

const NUM_RANDOM: usize = 1_000;

fn save_numbers_to_file(numbers: &[u64], filepath: &str) {
    let mut file = File::create(filepath).expect("failed to create output file");

    for &num in numbers {
        writeln!(file, "{}", num).expect("failed to write number");
    }

    println!("Random numbers written to: {}", filepath);
}

fn main() {
    let mut rng = Sphur::new_seeded(0x9e3779b97f4a7c15);
    let mut numbers = vec![0u64; NUM_RANDOM];

    for val in numbers.iter_mut() {
        *val = rng.gen_u64();
    }

    save_numbers_to_file(&numbers, "./target/prngs.txt");
    println!("Done ✅");
}
