use crate::cache::ThreeLevelCache;
use crate::config::ModelConfig;
use crate::gate::GateOutput;
use std::sync::Arc;

pub struct LayerPrefetcher {
    pub config: Arc<ModelConfig>,
    pub lookahead: usize,
}

impl LayerPrefetcher {
    pub fn new(config: Arc<ModelConfig>, lookahead: usize) -> Self {
        Self { config, lookahead }
    }

    pub fn predict_next_experts_from_indices(
        &self,
        indices: &[i32],
        topk: usize,
    ) -> Vec<usize> {
        let mut experts = Vec::new();
        for &idx in indices {
            if idx >= 0 && (idx as usize) < self.config.n_routed_experts {
                if !experts.contains(&(idx as usize)) {
                    experts.push(idx as usize);
                }
            }
            if experts.len() >= topk {
                break;
            }
        }
        experts
    }

    pub fn predict_next_experts(
        &self,
        current_layer: usize,
        gate_output: &GateOutput,
    ) -> Vec<usize> {
        let indices_host = gate_output.indices.to_host().unwrap();
        let indices: &[i32] = bytemuck::cast_slice(&indices_host.data);
        let result = self.predict_next_experts_from_indices(indices, self.config.num_experts_per_tok);

        let _ = current_layer;

        result
    }

    pub fn prefetch_to_gpu(
        &self,
        layer_id: usize,
        expert_ids: &[usize],
        cache: &mut ThreeLevelCache,
    ) -> Vec<usize> {
        let mut prefetched = Vec::new();

        for &expert_id in expert_ids {
            if !cache.gpu.contains(layer_id, expert_id) {
                if let Some(weights) = cache.gpu.get(layer_id, expert_id) {
                    let _ = weights;
                    prefetched.push(expert_id);
                }
            }
        }

        prefetched
    }
}
