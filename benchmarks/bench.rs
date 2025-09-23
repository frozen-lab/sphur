use sphur::Sphur;
use std::time::Instant;

const NUM_RANDOM: usize = 10_000;
const NUM_BINS: usize = 256;

fn bench_u128(rng: &mut Sphur) -> f64 {
    let mut buf = [0u128; NUM_RANDOM];
    let start = Instant::now();

    for val in buf.iter_mut() {
        *val = rng.gen_u128();
    }

    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    NUM_RANDOM as f64 / elapsed_us
}

fn bench_u64(rng: &mut Sphur) -> f64 {
    let mut buf = [0u64; NUM_RANDOM];
    let start = Instant::now();

    for val in buf.iter_mut() {
        *val = rng.gen_u64();
    }

    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    NUM_RANDOM as f64 / elapsed_us
}

fn bench_u32(rng: &mut Sphur) -> f64 {
    let mut buf = [0u32; NUM_RANDOM];
    let start = Instant::now();

    for val in buf.iter_mut() {
        *val = rng.gen_u32();
    }

    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    NUM_RANDOM as f64 / elapsed_us
}

fn bench_bool(rng: &mut Sphur) -> f64 {
    let mut buf = [false; NUM_RANDOM];
    let start = Instant::now();

    for val in buf.iter_mut() {
        *val = rng.gen_bool();
    }

    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    NUM_RANDOM as f64 / elapsed_us
}

fn bench_batch(rng: &mut Sphur) -> f64 {
    let mut total = 0;
    let start = Instant::now();
    let mut iterations = NUM_RANDOM / (16 * 2);

    while iterations > 0 {
        let batch = rng.gen_batch();
        total += batch.len();
        iterations -= 1;
    }

    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    total as f64 / elapsed_us
}

fn benchmark_randomness(numbers: &[u64]) -> (f64, f64) {
    // ▶ Histogram / uniformity (Chi-squared)
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

    (chi2, autocorr)
}

fn print_system_info() {
    let os = "Linux";
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

    // Detect SIMD extensions from flags
    let mut simd_features = vec!["sse2"]; // baseline on x86_64
    let f = flags.split_whitespace().collect::<Vec<_>>();

    if f.contains(&"avx") {
        simd_features.push("avx");
    }

    if f.contains(&"avx2") {
        simd_features.push("avx2");
    }

    let simd_info = simd_features.join(", ");

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
    println!("| SIMD Extensions | {:<30}   |", simd_info);
    println!("| RAM (GB)        | {:>28.2} |", total_mem_gb);
}

fn main() {
    println!("## Benchmakrs");

    let mut rng = Sphur::new_seeded(0x9e3779b97f4a7c15);

    let throughput_u128 = bench_u128(&mut rng);
    let throughput_u64 = bench_u64(&mut rng);
    let throughput_u32 = bench_u32(&mut rng);
    let throughput_bool = bench_bool(&mut rng);
    let throughput_batch = bench_batch(&mut rng);

    // system info
    print_system_info();

    println!("\n");
    println!("| API            | Throughput (numbers/µs) |");
    println!("|:--------------:|:-----------------------:|");
    println!("| gen_u128       |  {:>22.2} |", throughput_u128);
    println!("| gen_u64        |  {:>22.2} |", throughput_u64);
    println!("| gen_u32        |  {:>22.2} |", throughput_u32);
    println!("| gen_bool       |  {:>22.2} |", throughput_bool);
    println!("| gen_batch (32) |  {:>22.2} |", throughput_batch);

    rng = Sphur::new_seeded(0x9e3779b97f4a7c15);
    let mut numbers = vec![0u64; NUM_RANDOM];

    for val in numbers.iter_mut() {
        *val = rng.gen_u64();
    }

    let (chi2, autocorr) = benchmark_randomness(&numbers);

    println!("\n");
    println!("| Metric          | Value       |");
    println!("|:---------------:|:-----------:|");
    println!("| Chi-squared     | {:>10.2}  |", chi2);
    println!("| Autocorrelation | {:>10.5}  |", autocorr);
}
