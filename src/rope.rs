use crate::config::ModelConfig;

pub struct RopeCache {
    pub freqs_cos: Vec<f32>,
    pub freqs_sin: Vec<f32>,
    pub seqlen: usize,
    pub dim: usize,
}

impl RopeCache {
    pub fn precompute(config: &ModelConfig, seqlen: usize, layer_id: usize) -> Self {
        let dim = config.qk_rope_head_dim;
        let base = config.rope_theta_for_layer(layer_id);
        let factor = config.rope_factor_for_layer(layer_id);
        let original_seq_len = config.original_seq_len_for_layer(layer_id);
        let beta_fast = config.beta_fast();
        let beta_slow = config.beta_slow();

        let half_dim = dim / 2;
        let mut freqs = Vec::with_capacity(half_dim);

        for k in 0..half_dim {
            let f = 1.0 / base.powf(2.0 * k as f64 / dim as f64);
            freqs.push(f);
        }

        if original_seq_len > 0 && factor > 1.0 {
            let low = find_correction_dim(beta_fast as f64, dim as f64, base, original_seq_len as f64);
            let high = find_correction_dim(beta_slow as f64, dim as f64, base, original_seq_len as f64);
            let low = low.floor().max(0.0) as usize;
            let high = high.ceil().min((dim - 1) as f64) as usize;

            for k in 0..half_dim {
                let smooth = if low == high {
                    if k <= low { 1.0 } else { 0.0 }
                } else {
                    let linear = (k as f64 - low as f64) / (high as f64 - low as f64);
                    1.0 - linear.clamp(0.0, 1.0)
                };
                freqs[k] = freqs[k] / factor * (1.0 - smooth) + freqs[k] * smooth;
            }
        }

        let mut cos_data = Vec::with_capacity(seqlen * half_dim);
        let mut sin_data = Vec::with_capacity(seqlen * half_dim);

        for t in 0..seqlen {
            for k in 0..half_dim {
                let angle = t as f64 * freqs[k];
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
        }

        Self {
            freqs_cos: cos_data,
            freqs_sin: sin_data,
            seqlen,
            dim,
        }
    }

    pub fn get_slice(&self, start_pos: usize, len: usize) -> (&[f32], &[f32]) {
        let half_dim = self.dim / 2;
        let cos_start = start_pos * half_dim;
        let sin_start = start_pos * half_dim;
        let cos_end = cos_start + len * half_dim;
        let sin_end = sin_start + len * half_dim;
        (
            &self.freqs_cos[cos_start..cos_end],
            &self.freqs_sin[sin_start..sin_end],
        )
    }
}

fn find_correction_dim(num_rotations: f64, dim: f64, base: f64, max_seq_len: f64) -> f64 {
    dim * (max_seq_len / (num_rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
}
