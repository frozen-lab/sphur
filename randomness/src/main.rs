use sphur::Sphur;
use statrs::statistics::{Data, Distribution};

fn main() {
    const N: usize = 1_000_000;

    let total = N as f64;
    let rng = Sphur::new_seeded(0x1234);

    let mut samples = Vec::with_capacity(N);
    let mut hist = [0usize; 256];

    for _ in 0..N {
        samples.push(rng.next_u64() as f64 / u64::MAX as f64);
    }

    let data = Data::new(samples.clone());
    let mean = data.mean().unwrap_or(f64::NAN);
    let var = data.variance().unwrap_or(f64::NAN);

    for &v in &samples {
        let idx = (v * 255.0) as usize;
        hist[idx] += 1;
    }

    let entropy: f64 = hist
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum();

    println!("Stats:");
    println!("  mean     : {:.6}", mean);
    println!("  variance : {:.6}", var);
    println!("  entropy  : {:.3} bits", entropy);
    println!();
    println!("(expected ~mean=0.5, var=0.0833, entropy≈8 bits)");
}
