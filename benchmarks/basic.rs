///
/// Run bench using,
///
/// ```sh
/// sudo chrt -f 99 taskset -c 2 cargo bench --bench basic
/// ```
use sphur::Sphur;
use std::hint::black_box;
use std::time::Instant;
use std::{env, fs, io::Write, path::PathBuf};

const BATCH_SIZE: usize = 1024;

fn bench_next_u64(rng: &mut Sphur, iter: usize) -> f64 {
    let start = Instant::now();

    for _ in 0..iter {
        black_box(rng.next_u64());
    }

    let dur = start.elapsed().as_nanos();

    dur as f64 / iter as f64
}

fn bench_next_u32(rng: &mut Sphur, iter: usize) -> f64 {
    let start = Instant::now();

    for _ in 0..iter {
        black_box(rng.next_u32());
    }

    let dur = start.elapsed().as_nanos();

    dur as f64 / iter as f64
}

fn bench_batch_u64(rng: &mut Sphur, iter: usize) -> f64 {
    let mut buf = [0u64; BATCH_SIZE];
    let start = Instant::now();

    for _ in 0..(iter / BATCH_SIZE) {
        black_box(rng.batch_u64(&mut buf));
    }

    let dur = start.elapsed().as_nanos();

    dur as f64 / iter as f64
}

fn bench_batch_u32(rng: &mut Sphur, iter: usize) -> f64 {
    let mut buf = [0u32; BATCH_SIZE];
    let start = Instant::now();

    for _ in 0..(iter / BATCH_SIZE) {
        black_box(rng.batch_u32(&mut buf));
    }

    let dur = start.elapsed().as_nanos();

    dur as f64 / iter as f64
}

fn save_report(report: &str) -> std::io::Result<()> {
    let dir = PathBuf::from("./.local");

    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }

    let file_path = dir.join("bench.md");
    let mut file = fs::File::create(file_path)?;

    file.write_all(report.as_bytes())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);

    let noise = 0.2_f64;
    let mut rng = Sphur::new_seeded(0xDEADBEEF_u64);

    // warmup
    for _ in 0..1_000 {
        black_box(rng.next_u64());
    }

    let t_u64 = bench_next_u64(&mut rng, iters);
    let t_u32 = bench_next_u32(&mut rng, iters * 2);

    let t_batch_u64 = bench_batch_u64(&mut rng, iters);
    let t_batch_u32 = bench_batch_u32(&mut rng, iters * 2);

    let report = format!(
        r#"
## Benchmarks

| Function         | Time (ns/value)     | Throughput (vals/sec) |
|:----------------:|:-------------------:|:---------------------:|
| next_u64         | {:>14.4} ±{noise} | {:>21.2} |
| next_u32         | {:>14.4} ±{noise} | {:>21.2} |
| fill_u64_batch   | {:>14.4} ±{noise} | {:>21.2} |
| fill_u32_batch   | {:>14.4} ±{noise} | {:>21.2} |
"#,
        t_u64,
        1e9 / t_u64,
        t_u32,
        1e9 / t_u32,
        t_batch_u64,
        1e9 / t_batch_u64,
        t_batch_u32,
        1e9 / t_batch_u32
    );

    if let Err(e) = save_report(&report) {
        eprintln!("Failed to save report: {e}");
    }
}
