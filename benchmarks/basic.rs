use sphur::Sphur;
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 10_000_000;
const BATCH_SIZE: usize = 1024;

fn bench_next_u64(rng: &mut Sphur) -> f64 {
    let mut acc = 0u64;
    let start = Instant::now();

    for _ in 0..ITERS {
        acc = acc.wrapping_add(black_box(rng.next_u64()));
    }

    let dur = start.elapsed().as_nanos();
    black_box(acc);

    dur as f64 / ITERS as f64
}

fn bench_next_u32(rng: &mut Sphur) -> f64 {
    let mut acc = 0u32;
    let start = Instant::now();
    for _ in 0..ITERS {
        acc = acc.wrapping_add(black_box(rng.next_u32()));
    }
    let dur = start.elapsed().as_nanos();
    black_box(acc);

    dur as f64 / ITERS as f64
}

fn bench_batch_u64(rng: &mut Sphur) -> f64 {
    let mut buf = [0u64; BATCH_SIZE];
    let mut total = 0u128;
    let start = Instant::now();
    for _ in 0..(ITERS / BATCH_SIZE) {
        rng.batch_u64(&mut buf);
        total = total.wrapping_add(buf[0] as u128);
    }
    let dur = start.elapsed().as_nanos();
    black_box(total);

    dur as f64 / ITERS as f64
}

fn bench_batch_u32(rng: &mut Sphur) -> f64 {
    let mut buf = [0u32; BATCH_SIZE];
    let mut total = 0u128;
    let start = Instant::now();
    for _ in 0..(ITERS / BATCH_SIZE) {
        rng.batch_u32(&mut buf);
        total = total.wrapping_add(buf[0] as u128);
    }
    let dur = start.elapsed().as_nanos();
    black_box(total);

    dur as f64 / ITERS as f64
}

fn main() {
    let mut rng = Sphur::new_seeded(0xDEADBEEF_u64);

    // warmup
    for _ in 0..1_000 {
        black_box(rng.next_u64());
    }

    let t_u64 = bench_next_u64(&mut rng);
    let t_u32 = bench_next_u32(&mut rng);
    let t_batch_u64 = bench_batch_u64(&mut rng);
    let t_batch_u32 = bench_batch_u32(&mut rng);

    // throughput = 1e9 / ns_per_value
    let to_tps = |ns: f64| 1e9 / ns;

    let report = format!(
        r#"
## Benchmarks

| Function        | Time (ns/value)     | Throughput (vals/sec) |
|:---------------:|:-------------------:|:---------------------:|
| next_u64        | {:>15.4}            | {:>20.2}              |
| next_u32        | {:>15.4}            | {:>20.2}              |
| fill_u64_batch  | {:>15.4}            | {:>20.2}              |
| fill_u32_batch  | {:>15.4}            | {:>20.2}              |
"#,
        t_u64,
        to_tps(t_u64),
        t_u32,
        to_tps(t_u32),
        t_batch_u64,
        to_tps(t_batch_u64),
        t_batch_u32,
        to_tps(t_batch_u32)
    );

    println!("{report}");
}
