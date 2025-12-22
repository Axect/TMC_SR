use peroxide::fuga::*;
use rayon::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressIterator, ProgressBar};

#[allow(non_snake_case)]
fn main() {
    let N_EVENTS = 100_000;
    let T = 0.25;
    let N_vec = (1 ..=10).map(|x| x * 10).collect::<Vec<usize>>();
    let seed_list = (0 .. N_vec.len()).map(|x| 42 + 7 * x as u64).collect::<Vec<u64>>();

    let results: Vec<TMCResult> = N_vec.par_iter()
        .progress_with(ProgressBar::new(N_vec.len() as u64))
        .zip(seed_list.par_iter())
        .map(|(&N, &seed)| {
            TMCResult::generate(N_EVENTS, T, N, seed)
        })
        .collect();

    // Analyze Results
    // 1) Save momentum samples (N=20) to Parquet
    //let sample = &results[1]; // N=20
    //let mut px = Vec::with_capacity(N_EVENTS * sample.N);
    //let mut py = Vec::with_capacity(N_EVENTS * sample.N);
    //for event in sample.samples.iter() {
    //    for p in event.iter() {
    //        px.push(p.px);
    //        py.push(p.py);
    //    }
    //}
    //let mut df = DataFrame::new(vec![]);
    //df.push("px", Series::new(px));
    //df.push("py", Series::new(py));
    //df.write_parquet("data/samples_N20.parquet", SNAPPY).unwrap();

    // 2) Compute c2{2} with error bars and compare with approximation
    let n_batches = 50; // Number of batches for error estimation
    let c2_2_results: Vec<(f64, f64)> = results.iter()
        .progress_with(ProgressBar::new(N_vec.len() as u64))
        .map(|res| res.c2_2_with_error(n_batches))
        .collect();
    let c2_2_vec: Vec<f64> = c2_2_results.iter().map(|(m, _)| *m).collect();
    let c2_2_err_vec: Vec<f64> = c2_2_results.iter().map(|(_, e)| *e).collect();
    let c2_2_approx_vec = N_vec.iter()
        .map(|&N| c2_2_approx(N))
        .collect::<Vec<f64>>();
    let mut df_c2 = DataFrame::new(vec![]);
    let N_vec = N_vec.iter().map(|&n| n as i32).collect::<Vec<i32>>();
    df_c2.push("N", Series::new(N_vec));
    df_c2.push("c2_2", Series::new(c2_2_vec));
    df_c2.push("c2_2_err", Series::new(c2_2_err_vec));
    df_c2.push("c2_2_approx", Series::new(c2_2_approx_vec));
    df_c2.print();
    df_c2.write_parquet("data/c2_2_results.parquet", SNAPPY).unwrap();
}

#[allow(non_snake_case)]
struct TMCResult {
    samples: Vec<Vec<Momentum>>,
    N: usize,
}

impl TMCResult {
    #[allow(non_snake_case)]
    fn generate(n_events: usize, T: f64, N: usize, seed: u64) -> Self {
        let mut rng = stdrng_from_seed(seed);
        let sampler = TMCSampler::new(T, N);
        let samples = sampler.sample(n_events, &mut rng);
        TMCResult {
            samples,
            N,
        }
    }

    #[allow(non_snake_case)]
    fn c2_2_per_event(&self) -> Vec<f64> {
        let N_f64 = self.N as f64;
        self.samples.par_iter()
            .map(|event| {
                // Q2 = sum( exp( i * 2 * phi ) )
                let mut qx = 0.0;
                let mut qy = 0.0;

                for p in event.iter() {
                    let phi = p.phi();
                    qx += (2.0 * phi).cos();
                    qy += (2.0 * phi).sin();
                }
                let q_sq = qx * qx + qy * qy;
                (q_sq - N_f64) / (N_f64 * (N_f64 - 1.0))
            })
            .collect()
    }

    #[allow(non_snake_case)]
    fn c2_2(&self) -> f64 {
        let c2_2_values = self.c2_2_per_event();
        c2_2_values.iter().sum::<f64>() / c2_2_values.len() as f64
    }

    /// Returns (mean, std_error) of c2_2 using batch resampling
    #[allow(non_snake_case)]
    fn c2_2_with_error(&self, n_batches: usize) -> (f64, f64) {
        let c2_2_values = self.c2_2_per_event();
        let n = c2_2_values.len();
        let batch_size = n / n_batches;

        // Compute c2_2 for each batch
        let batch_means: Vec<f64> = (0..n_batches)
            .into_par_iter()
            .map(|i| {
                let start = i * batch_size;
                let end = if i == n_batches - 1 { n } else { (i + 1) * batch_size };
                let batch = &c2_2_values[start..end];
                batch.iter().sum::<f64>() / batch.len() as f64
            })
            .collect();

        // Calculate mean and standard error
        let mean = batch_means.iter().sum::<f64>() / n_batches as f64;
        let variance = batch_means.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n_batches - 1) as f64;
        let std_error = (variance / n_batches as f64).sqrt();

        (mean, std_error)
    }
}

#[allow(non_snake_case)]
fn c2_2_approx(N: usize) -> f64 {
    let N_f64 = N as f64;
    1f64 / (2f64 * (N_f64 - 2f64).powi(2))
}

#[derive(Debug, Copy, Clone)]
struct Momentum {
    px: f64,
    py: f64,
}

impl Momentum {
    fn new(px: f64, py: f64) -> Self {
        Momentum { px, py }
    }

    fn norm_squared(&self) -> f64 {
        self.px * self.px + self.py * self.py
    }

    fn phi(&self) -> f64 {
        self.py.atan2(self.px)
    }
}

#[allow(non_snake_case)]
struct TMCSampler {
    T: f64,
    N: usize,
}

impl TMCSampler {
    #[allow(non_snake_case)]
    fn new(T: f64, N: usize) -> Self {
        TMCSampler { T, N }
    }

    fn unconstrained_sample<R: Rng + Clone>(&self, n_samples: usize, rng: &mut R) -> Vec<Momentum> {
        let gamma = Gamma(2f64, 1f64 / self.T);
        let uniform = Uniform(0f64, 2f64 * std::f64::consts::PI);
        let p_mag = gamma.sample_with_rng(rng, n_samples);
        let angles: Vec<f64> = uniform.sample_with_rng(rng, n_samples);
        p_mag
            .into_iter()
            .zip(angles.into_iter())
            .map(|(p, theta)| Momentum::new(p * theta.cos(), p * theta.sin()))
            .collect()
    }

    #[allow(non_snake_case)]
    fn generate_one_event<R: Rng + Clone>(&self, rng: &mut R) -> Vec<Momentum> {
        let max_prob_val = self.T / std::f64::consts::E;
        loop {
            // Generate N-1 particles from Gamma distribution
            let mut particles = self.unconstrained_sample(self.N-1, rng);
            let mut px_sum = 0f64;
            let mut py_sum = 0f64;
            for &p in particles.iter() {
                px_sum += p.px;
                py_sum += p.py;
            }
            
            // Momentum of Nth particle to enforce total momentum conservation
            let p_N = Momentum::new(-px_sum, -py_sum);
            let p_N_mag = p_N.norm_squared().sqrt();

            // Accept or reject by Boltzmann distribution
            let current_kernel = p_N_mag * (-p_N_mag / self.T).exp();
            let acceptance_ratio = current_kernel / max_prob_val;
            let uniform = Uniform(0f64, 1f64);
            let u: f64 = uniform.sample_with_rng(rng, 1)[0];
            if u < acceptance_ratio {
                particles.push(p_N);
                return particles;
            }
        }
    }

    fn sample<R: Rng + Clone>(&self, n_events: usize, rng: &mut R) -> Vec<Vec<Momentum>> {
        (0..n_events)
            .into_iter()
            .map(|_| self.generate_one_event(rng))
            .collect()
    }
}
