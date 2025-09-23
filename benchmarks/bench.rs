use sphur::Sphur;
use std::time::Instant;

const NUM_RANDOM: usize = 10_000;
const NUM_ITER: usize = 1_000;
const SEED: u64 = 0x9e3779b97f4a7c15;

fn bench_bits<F>(mut rng_func: F, bits_per_call: usize) -> f64
where
    F: FnMut() -> (),
{
    let mut results = Vec::with_capacity(NUM_ITER);

    // Warm-up
    for _ in 0..10_000 {
        rng_func();
    }

    for _ in 0..NUM_ITER {
        let start = Instant::now();
        let mut bits_generated = 0usize;

        for _ in 0..NUM_RANDOM {
            rng_func();
            bits_generated += bits_per_call;
        }

        let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
        let bits_per_us = bits_generated as f64 / elapsed_us;
        results.push(bits_per_us);
    }

    results.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // median
    results[NUM_ITER / 2]
}

fn bench_throughput() {
    let bench_u128 = bench_bits(
        || {
            let mut rng = Sphur::new_seeded(SEED);
            let _ = rng.gen_u128();
        },
        128,
    );

    const BATCH_BITS: usize = 64 * 32;
    let bench_batch = bench_bits(
        || {
            let mut rng = Sphur::new_seeded(SEED);
            let _ = rng.gen_batch();
        },
        BATCH_BITS,
    );

    let bench_u64 = bench_bits(
        || {
            let mut rng = Sphur::new_seeded(SEED);
            let _ = rng.gen_u64();
        },
        64,
    );

    let bench_u32 = bench_bits(
        || {
            let mut rng = Sphur::new_seeded(SEED);
            let _ = rng.gen_u32();
        },
        32,
    );

    println!("\n");
    println!("| API            | Throughput (bits/µs) | Throughput (numbers/µs) |");
    println!("|:--------------:|:--------------------:|:-----------------------:|");
    println!(
        "| gen_u128       | {:>20.2} |  {:>22.2} |",
        bench_u128,
        bench_u128 / 128.0
    );
    println!(
        "| gen_u64        | {:>20.2} |  {:>22.2} |",
        bench_u64,
        bench_u64 / 64.0
    );
    println!(
        "| gen_u32        | {:>20.2} |  {:>22.2} |",
        bench_u32,
        bench_u32 / 32.0
    );
    println!(
        "| gen_batch (32) | {:>20.2} |  {:>22.2} |",
        bench_batch,
        bench_batch / 64.0
    );
}

fn bench_randomness() {
    const NUM_BINS: usize = 256;
    const NUM_ITER: usize = 10;

    let mut chi2_results = Vec::with_capacity(NUM_ITER);
    let mut autocorr_results = Vec::with_capacity(NUM_ITER);

    for _ in 0..NUM_ITER {
        let mut rng = Sphur::new_seeded(SEED);
        let mut numbers = vec![0u64; NUM_RANDOM];

        // generate numbers
        for val in numbers.iter_mut() {
            *val = rng.gen_u64();
        }

        // ▶ Histogram / uniformity (Chi-squared)
        let mut bins = vec![0usize; NUM_BINS];

        for &num in &numbers {
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

        chi2_results.push(chi2);

        // ▶ Autocorrelation (lag 1)
        let mean: f64 =
            numbers.iter().copied().map(|x| x as f64).sum::<f64>() / numbers.len() as f64;

        let mut num_acc = 0.0;
        let mut den_acc = 0.0;

        for i in 0..numbers.len() - 1 {
            let x = numbers[i] as f64 - mean;
            let y = numbers[i + 1] as f64 - mean;

            num_acc += x * y;
            den_acc += x * x;
        }

        let autocorr = num_acc / den_acc;

        autocorr_results.push(autocorr);
    }

    chi2_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
    autocorr_results.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_chi2 = chi2_results[NUM_ITER / 2];
    let median_autocorr = autocorr_results[NUM_ITER / 2];

    println!("\n");
    println!("| Metric          | Value       |");
    println!("|:---------------:|:-----------:|");
    println!("| Chi-squared     | {:>10.2}  |", median_chi2);
    println!("| Autocorrelation | {:>10.5}  |", median_autocorr);
}

fn fetch_system_info() {
    let os = std::env::consts::OS;
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").expect("Failed to read /proc/cpuinfo");

    let mut cpu = "Unknown CPU".to_string();
    let mut flags = String::new();

    for line in cpuinfo.lines() {
        if line.starts_with("model name") && cpu == "Unknown CPU" {
            if let Some(pos) = line.find(':') {
                cpu = line[(pos + 1)..].trim().to_string();
            }
        }

        if line.starts_with("flags") && flags.is_empty() {
            if let Some(pos) = line.find(':') {
                flags = line[(pos + 1)..].trim().to_string();
            }
        }
    }

    let isa: String = {
        let f = flags.split_whitespace().collect::<Vec<_>>();

        if f.contains(&"avx2") {
            "avx2".to_string()
        } else if f.contains(&"avx") {
            "avx".to_string()
        } else {
            "sse2".to_string()
        }
    };

    let meminfo = std::fs::read_to_string("/proc/meminfo").expect("Failed to read /proc/meminfo");
    let mut total_mem_kb: u64 = 0;

    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 2 {
                total_mem_kb = parts[1].parse::<u64>().expect("Failed to parse MemTotal");
            }

            break;
        }
    }

    let total_mem_gb = total_mem_kb as f64 / (1024.0 * 1024.0);

    println!("\n");
    println!("| System Info     | Value                          |");
    println!("|:---------------:|:------------------------------:|");
    println!("| OS              | {:<30}   |", os);
    println!("| CPU             | {:<30}   |", cpu);
    println!("| SIMD ISA        | {:<30}   |", isa);
    println!("| RAM (GB)        | {:>28.2} |", total_mem_gb);
}

fn main() {
    println!("## Benchmarks");

    fetch_system_info();
    bench_throughput();
    bench_randomness();
}
