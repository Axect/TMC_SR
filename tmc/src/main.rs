use indicatif::{ProgressBar, ProgressStyle};
use peroxide::fuga::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let mut rng = stdrng_from_seed(42);
    let sampler = TMCSampler::new(0.25, 20, 1e-8);  // stricter tolerance
    let samples = sampler.sample(10000, &mut rng);
    println!("Generated {} samples", samples.len());

    // Momentum Check
    let px: Vec<f64> = samples.iter().map(|m| m.data[0]).collect();
    let py: Vec<f64> = samples.iter().map(|m| m.data[1]).collect();
    let px_mean: f64 = px.par_iter().sum::<f64>() / samples.len() as f64;
    let py_mean: f64 = py.par_iter().sum::<f64>() / samples.len() as f64;
    println!("Mean px: {}, Mean py: {}", px_mean, py_mean);

    // c2 correlation
    
    // Save to Parquet
    let mut df = DataFrame::new(vec![]);
    df.push("px", Series::new(px));
    df.push("py", Series::new(py));
    df.print();
    df.write_parquet("data/samples.parquet", SNAPPY).unwrap();
}

#[derive(Debug, Clone)]
struct Momentum<const D: usize> {
    data: [f64; D],
}

impl<const D: usize> Momentum<D> {
    fn norm_squared(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }
}

#[allow(non_snake_case)]
struct TMCSampler {
    T: f64,
    N: usize,
    p_F: f64,
    p2_F: f64,
    tol: f64,
}

impl TMCSampler {
    #[allow(non_snake_case)]
    fn new(T: f64, N: usize, tol: f64) -> Self {
        let p_F = 2f64 * T;
        let p2_F = 6f64 * T.powi(2);
        TMCSampler { T, N, p_F, p2_F, tol }
    }

    fn unconstrained_sample<R: Rng + Clone>(&self, n_samples: usize, rng: &mut R) -> Vec<Momentum<2>> {
        let gamma = Gamma(2f64, 1f64 / self.T);
        let uniform = Uniform(0f64, 2f64* std::f64::consts::PI * 2f64);
        let p_mag = gamma.sample_with_rng(rng, n_samples);
        let angles: Vec<f64> = uniform.sample_with_rng(rng, n_samples);
        p_mag
            .into_iter()
            .zip(angles.into_iter())
            .map(|(p, theta)| Momentum {
                data: [p * theta.cos(), p * theta.sin()],
            })
            .collect()
    }

    fn sample<R: Rng + Clone>(&self, n_samples: usize, rng: &mut R) -> Vec<Momentum<2>> {
        let mut samples = Vec::with_capacity(n_samples);
        while samples.len() < n_samples {
            // Generate N-1 particles from Gamma distribution
            let p = self.unconstrained_sample(self.N-1, rng);
            let px = p.iter().map(|m| m.data[0]).collect::<Vec<f64>>();
            let py = p.iter().map(|m| m.data[1]).collect::<Vec<f64>>();
            let px_sum = px.par_iter().sum::<f64>();
            let py_sum = py.par_iter().sum::<f64>();
            
            // Momentum of Nth particle to enforce total momentum conservation
            let p_N = Momentum {
                data: [-px_sum, -py_sum],
            };
            let p_N_mag = (p_N.norm_squared()).sqrt();

            // Accept or reject by Boltzmann distribution
            let gamma = Gamma(2f64, 1f64 / self.T);
            let acceptance_prob = gamma.pdf(p_N_mag);
            let uniform = Uniform(0f64, 1f64);
            let u: f64 = uniform.sample_with_rng(rng, 1)[0];
            if u < acceptance_prob {
                let mut batch = p;
                batch.push(p_N);
                samples.extend(batch);
            }
        }
        samples
    }

    /// Parallel sampling: generates batches across multiple threads simultaneously
    fn sample_parallel(&self, n_samples: usize) -> Vec<Momentum<2>> {
        let n_threads = rayon::current_num_threads();
        let mut samples = Vec::with_capacity(n_samples);

        // Atomic counter for unique seeds across all threads and iterations
        let base_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let seed_counter = AtomicU64::new(base_seed);

        // Counters for statistics
        let total_trials = AtomicUsize::new(0);
        let accepted_count = AtomicUsize::new(0);

        // Setup progress bar
        let pb = ProgressBar::new(n_samples as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | Trials: {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        while samples.len() < n_samples {
            // Generate n_threads batches in parallel
            let accepted: Vec<Vec<Momentum<2>>> = (0..n_threads)
                .into_par_iter()
                .filter_map(|_| {
                    // Each thread gets a unique seed via atomic increment
                    let seed = seed_counter.fetch_add(1, Ordering::Relaxed);
                    let mut rng = smallrng_from_seed(seed);
                    let p = self.unconstrained_sample(self.N, &mut rng);

                    total_trials.fetch_add(1, Ordering::Relaxed);

                    // Check TMC constraint
                    let px_sum: f64 = p.iter().map(|m| m.data[0]).sum::<f64>() / self.N as f64;
                    let py_sum: f64 = p.iter().map(|m| m.data[1]).sum::<f64>() / self.N as f64;
                    let p_sum = px_sum.powi(2) + py_sum.powi(2);

                    if p_sum < self.tol {
                        accepted_count.fetch_add(1, Ordering::Relaxed);
                        Some(p)
                    } else {
                        None
                    }
                })
                .collect();

            // Add all accepted batches
            for batch in accepted {
                samples.extend(batch);
            }

            // Update progress bar
            let trials = total_trials.load(Ordering::Relaxed);
            let accepts = accepted_count.load(Ordering::Relaxed);
            let rate = if trials > 0 {
                100.0 * accepts as f64 / trials as f64
            } else {
                0.0
            };
            pb.set_position(samples.len().min(n_samples) as u64);
            pb.set_message(format!("{} (Accept: {:.2}%)", trials, rate));
        }

        pb.finish_with_message(format!(
            "{} (Accept: {:.2}%)",
            total_trials.load(Ordering::Relaxed),
            100.0 * accepted_count.load(Ordering::Relaxed) as f64
                / total_trials.load(Ordering::Relaxed) as f64
        ));

        samples.truncate(n_samples);
        samples
    }
}
