use sphur::Sphur;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const NUM_BINS: usize = 256;
const NUM_RANDOM: usize = 10_000;

// Benchmark throughput & store numbers in buffer
fn benchmark_throughput(rng: &mut Sphur, numbers: &mut [u64]) {
    let start = Instant::now();

    for val in numbers.iter_mut() {
        *val = rng.rand().expect("failed to generate rand");
    }

    let elapsed = start.elapsed();
    let elapsed_us = elapsed.as_secs_f64() * 1e6;
    let nums_per_us = numbers.len() as f64 / elapsed_us;

    println!(
        "Throughput: {:.2} random numbers per microsecond",
        nums_per_us
    );
}

// Benchmark randomness: histogram & autocorrelation
fn benchmark_randomness(numbers: &[u64]) {
    // ▶ Histogram / uniformity
    let mut bins = vec![0usize; NUM_BINS];

    for &num in numbers {
        bins[(num % NUM_BINS as u64) as usize] += 1;
    }

    let expected = numbers.len() as f64 / NUM_BINS as f64;
    let chi2: f64 = bins
        .iter()
        .map(|&count| {
            let diff = count as f64 - expected;
            diff * diff / expected
        })
        .sum();

    println!("Chi-squared (uniformity): {:.2}", chi2);

    // ▶ Autocorrelation (lag 1)
    let mean: f64 = numbers.iter().copied().map(|x| x as f64).sum::<f64>() / numbers.len() as f64;

    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..numbers.len() - 1 {
        let x = numbers[i] as f64 - mean;
        let y = numbers[i + 1] as f64 - mean;
        num += x * y;
        den += x * x;
    }

    let autocorr = num / den;
    println!("Autocorrelation (lag 1): {:.5}", autocorr);
}

// Save numbers to file for Python visualization
fn save_numbers_to_file(numbers: &[u64], filepath: &str) {
    let mut file = File::create(filepath).expect("failed to create output file");

    for &num in numbers {
        writeln!(file, "{}", num).expect("failed to write number");
    }

    println!("Random numbers written to: {}", filepath);
}

fn main() {
    println!("=== Sphur PRNG Benchmark ===");

    let mut rng = Sphur::new_seeded(0x9e3779b97f4a7c15).expect("failed to init sphur with seed");

    let mut numbers = vec![0u64; NUM_RANDOM];

    // Generate numbers & benchmark throughput
    benchmark_throughput(&mut rng, &mut numbers);

    // Benchmark randomness using the same numbers
    benchmark_randomness(&numbers);

    // Save numbers for Python plotting
    save_numbers_to_file(&numbers, "./target/prngs.txt");

    println!("Done ✅");
}
