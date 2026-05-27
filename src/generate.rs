use crate::model::Transformer;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use std::time::Instant;

/// 推理统计信息
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// 生成 token 数（不含 prompt）
    pub total_tokens: usize,
    /// 生成耗时（秒）
    pub elapsed_sec: f64,
    /// 每秒生成 token 数
    pub tokens_per_sec: f64,
    /// GPU 显存峰值（MB）
    pub gpu_peak_mb: f64,
    /// CPU 内存峰值（MB）
    pub cpu_peak_mb: f64,
}

/// 自回归生成配置
pub struct GenerateConfig {
    /// 最大生成 token 数
    pub max_new_tokens: usize,
    /// 采样温度（0 = greedy argmax）
    pub temperature: f32,
    /// 核采样阈值（1.0 = 不过滤）
    pub top_p: f32,
    /// 重复惩罚系数（1.0 = 无惩罚）
    pub repetition_penalty: f32,
    /// 重复惩罚滑动窗口大小
    pub repetition_window: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.6,
            top_p: 0.95,
            repetition_penalty: 1.0,
            repetition_window: 64,
        }
    }
}

/// 自回归生成器：封装模型前向传播 + 采样 + 分词
/// 维护 KV cache 位置计数器，支持多轮连续生成
pub struct Generator {
    model: Transformer,
    tokenizer: Tokenizer,
    config: GenerateConfig,
    /// 当前 KV cache 位置（即已处理的 token 数）
    position: usize,
}

impl Generator {
    pub fn new(model: Transformer, tokenizer: Tokenizer, config: GenerateConfig) -> Self {
        Self { model, tokenizer, config, position: 0 }
    }

    pub fn model(&mut self) -> &mut Transformer {
        &mut self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// 重置生成状态（清空 KV cache 位置），用于新对话
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// 自回归生成主循环
    /// 1. Prefill：一次性前向传播全部 prompt tokens
    /// 2. Decode：逐 token 前向传播 + 采样，直到 EOS 或达到最大长度
    /// callback 用于流式输出每个新生成的 token
    /// 返回 (completion_tokens, stats) 元组
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        callback: Option<&dyn Fn(&str)>,
    ) -> Result<(Vec<u32>, InferenceStats)> {
        let gen_start = Instant::now();
        let gpu_start_mb = self.get_gpu_memory_mb().unwrap_or(0.0);
        let cpu_start_kb = Self::get_cpu_memory_kb();

        let eos_id = self.tokenizer.eos_id();
        let prompt_len = prompt_tokens.len();
        let max_len = prompt_len + self.config.max_new_tokens;

        eprintln!("[prompt_tokens: {:?}]", prompt_tokens);

        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

        // Prefill 阶段：一次性处理全部 prompt，获取最后一个位置的 logits
        let logits = self.model.forward(prompt_tokens, self.position)?;
        self.position += prompt_len;

        {
            let logits_host = logits.to_host()?;
            let logits_f32: &[f32] = bytemuck::cast_slice(&logits_host.data);
            let mut indexed: Vec<(usize, f32)> = logits_f32.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("[logits prefill] top-5: {:?}", &indexed[..5.min(indexed.len())]);
        }

        // 从 prefill logits 采样第一个生成 token
        let mut next_token = self.sample_token(&logits, &all_tokens)?;

        if next_token == eos_id {
            all_tokens.push(next_token);
            let stats = self.compute_stats(&gen_start, gpu_start_mb, cpu_start_kb, 1);
            return Ok((all_tokens, stats));
        }

        all_tokens.push(next_token);
        if let Some(cb) = callback {
            let text = self.tokenizer.decode(&[next_token])?;
            cb(&text);
        }

        // Decode 阶段：逐 token 生成
        for _ in 1..self.config.max_new_tokens {
            let logits = self.model.forward(&[next_token], self.position)?;
            self.position += 1;

            next_token = self.sample_token(&logits, &all_tokens)?;
            all_tokens.push(next_token);

            if next_token == eos_id {
                break;
            }

            if let Some(cb) = callback {
                let text = self.tokenizer.decode(&[next_token])?;
                cb(&text);
            }

            if all_tokens.len() >= max_len {
                break;
            }
        }

        let total_tokens = all_tokens.len() - prompt_len;
        let stats = self.compute_stats(&gen_start, gpu_start_mb, cpu_start_kb, total_tokens);
        Ok((all_tokens, stats))
    }

    /// 计算推理统计信息
    fn compute_stats(&self, start: &Instant, gpu_start_mb: f64, cpu_start_kb: i64, total_tokens: usize) -> InferenceStats {
        let elapsed = start.elapsed().as_secs_f64();
        let tokens_per_sec = if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 };
        
        let gpu_peak_mb = self.get_gpu_memory_mb().unwrap_or(gpu_start_mb);
        let cpu_peak_kb = Self::get_cpu_memory_kb();
        let cpu_peak_mb = if cpu_peak_kb > cpu_start_kb { (cpu_peak_kb - cpu_start_kb) as f64 / 1024.0 } else { 0.0 };

        InferenceStats {
            total_tokens,
            elapsed_sec: elapsed,
            tokens_per_sec,
            gpu_peak_mb,
            cpu_peak_mb,
        }
    }

    /// 获取当前 GPU 显存使用量（MB）
    fn get_gpu_memory_mb(&self) -> Option<f64> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let result = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let used = total - free;
                return Some(used as f64 / (1024.0 * 1024.0));
            }
        }
        None
    }

    /// 获取当前进程 CPU 内存使用量（KB）
    fn get_cpu_memory_kb() -> i64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(status) = fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<i64>() {
                                return kb;
                            }
                        }
                    }
                }
            }
        }
        0
    }

    /// 从 logits 中采样一个 token
    /// 流程：温度缩放 → 重复惩罚 → softmax → top-p 过滤 → 随机采样
    fn sample_token(&self, logits: &crate::GpuTensor, generated: &[u32]) -> Result<u32> {
        // 将 GPU logits 拷贝到 CPU 进行采样
        let logits_host = logits.to_host()?;
        let logits_f32: &[f32] = bytemuck::cast_slice(&logits_host.data);

        let vocab_size = logits_f32.len();
        let mut probs = logits_f32.to_vec();

        // 温度 = 0 时直接取 argmax（greedy 解码）
        if self.config.temperature < 1e-5 {
            let best = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            return Ok(best.0 as u32);
        }

        // 温度缩放：logits / temperature
        for v in &mut probs {
            *v /= self.config.temperature;
        }

        // 重复惩罚：对滑动窗口内已生成的 token 降低概率
        if self.config.repetition_penalty > 1.0 {
            let window_start = generated.len().saturating_sub(self.config.repetition_window);
            for &tid in &generated[window_start..] {
                let idx = tid as usize;
                if idx < vocab_size {
                    if probs[idx] > 0.0 {
                        probs[idx] /= self.config.repetition_penalty;
                    } else {
                        probs[idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // 数值稳定的 softmax：减去最大值后取 exp
        let max_val = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut probs {
            *v = (*v - max_val).exp();
        }

        // Top-p（核采样）：保留累积概率达到 top_p 的最小 token 集合
        if self.config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            let sum: f32 = probs.iter().sum();
            let mut cumsum = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, &(_, p)) in indexed.iter().enumerate() {
                cumsum += p / sum;
                if cumsum > self.config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            let allowed: std::collections::HashSet<usize> =
                indexed[..cutoff].iter().map(|(i, _)| *i).collect();
            for (i, v) in probs.iter_mut().enumerate() {
                if !allowed.contains(&i) {
                    *v = 0.0;
                }
            }
        }

        // 归一化概率分布
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Ok(0u32);
        }
        for v in &mut probs {
            *v /= sum;
        }

        // 按概率分布随机采样
        let mut rng = rand::rng();
        let r: f32 = rand::Rng::random(&mut rng);
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return Ok(i as u32);
            }
        }
        Ok((vocab_size - 1) as u32)
    }
}
