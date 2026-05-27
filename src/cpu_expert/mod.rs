pub mod avx512;
pub mod kernel;
pub mod route;
pub mod tables;
pub mod tile_layout;

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

#[pyfunction]
fn is_avx512_supported() -> bool {
    avx512::is_avx512_supported()
}

#[pyfunction]
fn is_avx2_supported() -> bool {
    avx512::is_avx2_supported()
}

#[pyfunction]
fn init_tables(grid_u64: Vec<u64>, ksigns: Vec<u8>) {
    tables::init_tables(grid_u64, ksigns);
}

#[pyfunction]
fn is_tables_initialized() -> bool {
    tables::is_tables_initialized()
}

/// IQ2_XS weight stored in Rust, holding owned data for cache-friendly access.
#[pyclass]
struct Iq2XsWeight {
    inner: kernel::Iq2XsWeight,
}

/// IQ2_XXS weight (2.0625 bpw, no sub-block scales).
#[pyclass]
struct Iq2XxsWeight {
    inner: kernel::Iq2XxsWeight,
}

/// Q2_K weight (2.5625 bpw, 2-bit quantized + sub-block scales).
#[pyclass]
struct Q2KWeight {
    inner: kernel::Q2KWeight,
}

/// FP4 (e2m1) weight stored in Rust, packed format with per-block scale.
#[pyclass]
struct Fp4Weight {
    inner: kernel::Fp4Weight,
}

#[pymethods]
impl Fp4Weight {
    #[new]
    fn new<'py>(
        weight_packed: PyReadonlyArray1<'py, u8>,
        scale: PyReadonlyArray1<'py, u8>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let packed_vec = weight_packed.as_slice()?.to_vec();
        let scale_vec = scale.as_slice()?.to_vec();
        Ok(Self {
            inner: kernel::Fp4Weight::new(packed_vec, scale_vec, shape),
        })
    }

    fn size_bytes(&self) -> usize {
        kernel::QuantizedWeight::size_bytes(&self.inner)
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape
    }

    /// FP4 × F32 矩阵向量乘法（FMA 管线）
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_slice = x.as_slice()?;
        let result = kernel::QuantizedWeight::matvec(&self.inner, x_slice);
        Ok(PyArray1::from_vec(py, result))
    }
}

/// FP4 CPU expert FFN（gate/up/down 配对点积 + SwiGLU）
#[pyfunction]
fn cpu_expert_ffn_pair_fp4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Fp4Weight,
    up_weight: &Fp4Weight,
    down_weight: &Fp4Weight,
    route_weight: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::fp4_expert_ffn_pair(
        x_slice,
        &gate_weight.inner,
        &up_weight.inner,
        &down_weight.inner,
        route_weight,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// IQ2_XS 权重的 NHWC/Tile 内存布局，减少 cache miss。
#[pyclass]
struct Iq2XsTile {
    inner: tile_layout::Iq2XsTile,
}

#[pymethods]
impl Iq2XsWeight {
    #[new]
    fn new(d: Vec<f32>, qs: Vec<u16>, scales: Vec<u8>, shape: (usize, usize)) -> Self {
        Self {
            inner: kernel::Iq2XsWeight::new(d, qs, scales, shape),
        }
    }

    /// Create from numpy arrays - reads directly from numpy buffer, much faster than .tolist()
    #[staticmethod]
    fn from_numpy<'py>(
        d: PyReadonlyArray1<'py, f32>,
        qs: PyReadonlyArray1<'py, u16>,
        scales: PyReadonlyArray1<'py, u8>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let d_vec = d.as_slice()?.to_vec();
        let qs_vec = qs.as_slice()?.to_vec();
        let scales_vec = scales.as_slice()?.to_vec();
        Ok(Self {
            inner: kernel::Iq2XsWeight::new(d_vec, qs_vec, scales_vec, shape),
        })
    }
}

#[pymethods]
impl Iq2XxsWeight {
    #[new]
    fn new(d: Vec<f32>, qs: Vec<u16>, shape: (usize, usize)) -> Self {
        Self {
            inner: kernel::Iq2XxsWeight::new(d, qs, shape),
        }
    }

    /// Create from numpy arrays (zero-copy read)
    #[staticmethod]
    fn from_numpy<'py>(
        d: PyReadonlyArray1<'py, f32>,
        qs: PyReadonlyArray1<'py, u16>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let d_vec = d.as_slice()?.to_vec();
        let qs_vec = qs.as_slice()?.to_vec();
        Ok(Self {
            inner: kernel::Iq2XxsWeight::new(d_vec, qs_vec, shape),
        })
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape
    }

    fn size_bytes(&self) -> usize {
        kernel::QuantizedWeight::size_bytes(&self.inner)
    }

    /// IQ2_XXS 矩阵向量乘法
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_slice = x.as_slice()?;
        let result = kernel::QuantizedWeight::matvec(&self.inner, x_slice);
        Ok(PyArray1::from_vec(py, result))
    }
}

#[pymethods]
impl Q2KWeight {
    #[new]
    fn new(d: Vec<f32>, dmin: Vec<f32>, scales: Vec<u8>, qs: Vec<u8>, shape: (usize, usize)) -> Self {
        Self {
            inner: kernel::Q2KWeight::new(d, dmin, scales, qs, shape),
        }
    }

    /// Create from numpy arrays (zero-copy read)
    #[staticmethod]
    fn from_numpy<'py>(
        d: PyReadonlyArray1<'py, f32>,
        dmin: PyReadonlyArray1<'py, f32>,
        scales: PyReadonlyArray1<'py, u8>,
        qs: PyReadonlyArray1<'py, u8>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let d_vec = d.as_slice()?.to_vec();
        let dmin_vec = dmin.as_slice()?.to_vec();
        let scales_vec = scales.as_slice()?.to_vec();
        let qs_vec = qs.as_slice()?.to_vec();
        Ok(Self {
            inner: kernel::Q2KWeight::new(d_vec, dmin_vec, scales_vec, qs_vec, shape),
        })
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape
    }

    fn size_bytes(&self) -> usize {
        kernel::QuantizedWeight::size_bytes(&self.inner)
    }

    /// Q2_K 矩阵向量乘法
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_slice = x.as_slice()?;
        let result = kernel::QuantizedWeight::matvec(&self.inner, x_slice);
        Ok(PyArray1::from_vec(py, result))
    }
}

#[pymethods]
impl Iq2XsTile {
    /// 从 Iq2XsWeight 转换为 Tile 布局
    #[staticmethod]
    fn from_weight(weight: &Iq2XsWeight) -> Self {
        Self {
            inner: tile_layout::Iq2XsTile::from_weight(&weight.inner),
        }
    }

    /// 从 numpy 数组构建 Tile 布局
    #[staticmethod]
    fn from_numpy<'py>(
        d: PyReadonlyArray1<'py, f32>,
        qs: PyReadonlyArray1<'py, u16>,
        scales: PyReadonlyArray1<'py, u8>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let d_vec = d.as_slice()?.to_vec();
        let qs_vec = qs.as_slice()?.to_vec();
        let scales_vec = scales.as_slice()?.to_vec();
        Ok(Self {
            inner: tile_layout::Iq2XsTile::from_separate(&d_vec, &qs_vec, &scales_vec, shape),
        })
    }

    fn memory_usage_mb(&self) -> f64 {
        self.inner.memory_usage() as f64 / (1024.0 * 1024.0)
    }

    fn n_blocks(&self) -> usize {
        self.inner.n_blocks()
    }
}

/// CPU expert FFN with paired gate/up weights using Tile layout (zero-copy numpy input).
#[pyfunction]
fn cpu_expert_ffn_pair_tile<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XsTile,
    up_weight: &Iq2XsTile,
    down_weight: &Iq2XsTile,
    route_weight: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::cpu_expert_ffn_pair_tile(
        x_slice,
        &gate_weight.inner,
        &up_weight.inner,
        &down_weight.inner,
        route_weight,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// IQ2_XS matrix-vector product with Tile layout (zero-copy numpy input).
#[pyfunction]
fn iq2xs_matvec_tile<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    weight: &Iq2XsTile,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::iq2xs_matvec_tile(&weight.inner, x_slice);
    Ok(PyArray1::from_vec(py, result))
}

/// CPU expert FFN with paired gate/up weights (zero-copy numpy input).
#[pyfunction]
fn cpu_expert_ffn_pair<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XsWeight,
    up_weight: &Iq2XsWeight,
    down_weight: &Iq2XsWeight,
    route_weight: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::cpu_expert_ffn_pair(
        x_slice,
        &gate_weight.inner,
        &up_weight.inner,
        &down_weight.inner,
        route_weight,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// CPU expert FFN with fused gate_up weight (zero-copy numpy input).
#[pyfunction]
fn cpu_expert_ffn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_up_weight: &Iq2XsWeight,
    down_weight: &Iq2XsWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::cpu_expert_ffn(
        x_slice,
        &gate_up_weight.inner,
        &down_weight.inner,
        route_weight,
        swiglu_limit,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// 双专家 FFN（MoE top-2 场景，零拷贝 numpy 输入）。
#[pyfunction]
fn cpu_expert_ffn_pair_dual<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    expert_a_gate: &Iq2XsWeight,
    expert_a_up: &Iq2XsWeight,
    expert_a_down: &Iq2XsWeight,
    expert_b_gate: &Iq2XsWeight,
    expert_b_up: &Iq2XsWeight,
    expert_b_down: &Iq2XsWeight,
    route_weight_a: f32,
    route_weight_b: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::cpu_expert_ffn_pair_dual(
        x_slice,
        &expert_a_gate.inner, &expert_a_up.inner, &expert_a_down.inner,
        &expert_b_gate.inner, &expert_b_up.inner, &expert_b_down.inner,
        route_weight_a,
        route_weight_b,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// IQ2_XS matrix-vector product (zero-copy numpy input).
#[pyfunction]
fn iq2xs_matvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    weight: &Iq2XsWeight,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let n_rows = weight.inner.shape.0;
    let n_cols = weight.inner.shape.1;
    let result = kernel::iq2xs_matvec(&weight.inner, x_slice, n_rows, n_cols);
    Ok(PyArray1::from_vec(py, result))
}

#[pyclass]
struct CpuExpertRunner {
    inner: kernel::CpuExpertRunner,
}

#[pymethods]
impl CpuExpertRunner {
    #[new]
    fn new() -> Self {
        Self {
            inner: kernel::CpuExpertRunner::new(),
        }
    }

    fn add_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: &Iq2XsWeight,
        up_weight: &Iq2XsWeight,
        down_weight: &Iq2XsWeight,
    ) {
        self.inner.add_expert(
            layer_id,
            expert_id,
            gate_weight.inner.clone(),
            up_weight.inner.clone(),
            down_weight.inner.clone(),
        );
    }

    /// 添加专家到 protected 段（GPU 热专家同步专用，跳过 probation 准入）
    fn add_expert_protected(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: &Iq2XsWeight,
        up_weight: &Iq2XsWeight,
        down_weight: &Iq2XsWeight,
    ) {
        self.inner.add_expert_protected(
            layer_id,
            expert_id,
            gate_weight.inner.clone(),
            up_weight.inner.clone(),
            down_weight.inner.clone(),
        );
    }

    fn compute_expert<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: usize,
        expert_id: usize,
        x: PyReadonlyArray1<'py, f32>,
        route_weight: f32,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let x_slice = x.as_slice()?;
        let result = self.inner.compute_expert(layer_id, expert_id, x_slice, route_weight);
        match result {
            Some(vec) => Ok(Some(PyArray1::from_vec(py, vec))),
            None => Ok(None),
        }
    }

    /// 双专家 FFN（MoE top-2，使用常驻权重，零传输开销）
    fn compute_dual_expert<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: usize,
        expert_a: usize,
        expert_b: usize,
        x: PyReadonlyArray1<'py, f32>,
        route_weight_a: f32,
        route_weight_b: f32,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let x_slice = x.as_slice()?;
        let result = self.inner.compute_dual_expert(
            layer_id, expert_a, expert_b, x_slice, route_weight_a, route_weight_b,
        );
        match result {
            Some(vec) => Ok(Some(PyArray1::from_vec(py, vec))),
            None => Ok(None),
        }
    }

    fn has_expert(&self, layer_id: usize, expert_id: usize) -> bool {
        self.inner.has_expert(layer_id, expert_id)
    }

    /// 记录专家访问（不加载权重，只更新频率）
    fn record_access(&mut self, layer_id: usize, expert_id: usize) {
        self.inner.record_access(layer_id, expert_id);
    }

    fn remove_expert(&mut self, layer_id: usize, expert_id: usize) -> bool {
        self.inner.remove_expert(layer_id, expert_id)
    }

    fn expert_count(&self) -> usize {
        self.inner.expert_count()
    }

    fn memory_usage_mb(&self) -> f64 {
        self.inner.memory_usage() as f64 / (1024.0 * 1024.0)
    }

    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate()
    }

    fn set_protected_layer(&mut self, layer: Option<usize>) {
        self.inner.set_protected_layer(layer);
    }

    fn set_step_protected(&mut self, keys: std::collections::HashSet<(usize, usize)>) {
        self.inner.set_step_protected(keys);
    }

    fn clear_step_protected(&mut self) {
        self.inner.clear_step_protected();
    }

    fn save_freq(&self, path: &str) -> PyResult<()> {
        self.inner.save_freq(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn load_freq(&mut self, path: &str) -> PyResult<()> {
        self.inner.load_freq(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// 返回 protected 段的专家 key 列表
    fn protected_keys(&self) -> Vec<(usize, usize)> {
        self.inner.protected_keys()
    }

    /// 返回 probation 段的专家 key 列表
    fn probation_keys(&self) -> Vec<(usize, usize)> {
        self.inner.probation_keys()
    }

    /// L3 权重预热：将指定层的热专家权重数据预取到 L3 缓存
    fn warmup_layer(&self, layer_id: usize) {
        self.inner.warmup_layer(layer_id);
    }

    /// L3 权重预热（分层预加载版）：只预热 CPU 负责的 TopN+1 ~ TopN+M 专家
    ///
    /// Args:
    ///   layer_id: 要预热的层
    ///   gpu_keys: GPU SLRU 中已有的专家 key 集合
    ///   gpu_topn: GPU 负责的 Top-N 数量
    ///   warmup_m: CPU L3 预热的专家数量
    fn warmup_layer_targeted(
        &self,
        layer_id: usize,
        gpu_keys: std::collections::HashSet<(usize, usize)>,
        gpu_topn: usize,
        warmup_m: usize,
    ) {
        self.inner.warmup_layer_targeted(layer_id, &gpu_keys, gpu_topn, warmup_m);
    }

    /// 返回指定层的专家按频率降序排名列表
    ///
    /// 返回: Vec<(expert_id, frequency)>
    fn layer_freq_rank(&self, layer_id: usize) -> Vec<(usize, u64)> {
        self.inner.layer_freq_rank(layer_id)
    }

    /// 设置字节预算（兼容不同大小专家）
    fn set_bytes_budget(&mut self, protected_bytes: usize, probation_bytes: usize) {
        self.inner.set_bytes_budget(protected_bytes, probation_bytes);
    }

    /// 自动设置字节预算：根据系统可用内存和专家总量自适应
    fn auto_set_budget(&mut self, total_expert_bytes: usize, available_bytes: usize) {
        self.inner.auto_set_budget(total_expert_bytes, available_bytes);
    }
}

/// CPU FFN 引擎（IQ2_XXS+Q2_K 混合量化）
///
/// 封装专家权重存储 + SLRU 缓存管理 + FFN 计算。
/// Python 侧只需调用 add_expert + compute_ffn，无需关心权重细节。
#[pyclass]
struct CpuFfnEngineIq2xxsQ2k {
    inner: kernel::CpuFfnEngineIq2xxsQ2k,
}

#[pymethods]
impl CpuFfnEngineIq2xxsQ2k {
    #[new]
    fn new() -> Self {
        Self {
            inner: kernel::CpuFfnEngineIq2xxsQ2k::new(),
        }
    }

    /// 添加专家到缓存
    fn add_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: &Iq2XxsWeight,
        up_weight: &Iq2XxsWeight,
        down_weight: &Q2KWeight,
    ) {
        self.inner.add_expert(
            layer_id,
            expert_id,
            gate_weight.inner.clone(),
            up_weight.inner.clone(),
            down_weight.inner.clone(),
        );
    }

    /// 添加专家到 protected 段（热专家）
    fn add_expert_protected(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: &Iq2XxsWeight,
        up_weight: &Iq2XxsWeight,
        down_weight: &Q2KWeight,
    ) {
        self.inner.add_expert_protected(
            layer_id,
            expert_id,
            gate_weight.inner.clone(),
            up_weight.inner.clone(),
            down_weight.inner.clone(),
        );
    }

    /// 计算 FFN
    fn compute_ffn<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: usize,
        expert_id: usize,
        x: PyReadonlyArray1<'py, f32>,
        route_weight: f32,
        swiglu_limit: f32,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let x_slice = x.as_slice()?;
        let result = self.inner.compute_ffn(layer_id, expert_id, x_slice, route_weight, swiglu_limit);
        match result {
            Some(vec) => Ok(Some(PyArray1::from_vec(py, vec))),
            None => Ok(None),
        }
    }

    /// 记录专家访问（Gate 选中时调用，驱动预取）
    fn record_access(&mut self, layer_id: usize, expert_id: usize) {
        self.inner.record_access(layer_id, expert_id);
    }

    fn has_expert(&self, layer_id: usize, expert_id: usize) -> bool {
        self.inner.has_expert(layer_id, expert_id)
    }

    fn expert_count(&self) -> usize {
        self.inner.expert_count()
    }

    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate()
    }

    /// 返回 (hits, misses, protected_count, probation_count)
    fn stats(&self) -> (u64, u64, usize, usize) {
        self.inner.stats()
    }
}

/// 混合量化 FFN：IQ2_XXS gate/up + Q2_K down（零拷贝 numpy 输入）
#[pyfunction]
fn mixed_ffn_pair_iq2xxs_q2k<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::mixed_ffn_pair_iq2xxs_q2k(
        x_slice,
        &gate_weight.inner,
        &up_weight.inner,
        &down_weight.inner,
        route_weight,
        swiglu_limit,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// 混合量化 FFN（单线程顺序扫描 + 软件预取优化版）
#[pyfunction]
fn mixed_ffn_pair_iq2xxs_q2k_streaming<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::mixed_ffn_pair_iq2xxs_q2k_streaming(
        x_slice,
        &gate_weight.inner,
        &up_weight.inner,
        &down_weight.inner,
        route_weight,
        swiglu_limit,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// 6 专家并行 FFN（IQ2_XXS+Q2_K 混合量化）
#[pyfunction]
fn mixed_ffn_pair_iq2xxs_q2k_blocked_mt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let result = kernel::mixed_ffn_pair_iq2xxs_q2k_blocked_mt(
        x_slice,
        &gate_weight.inner,
        &up_weight.inner,
        &down_weight.inner,
        route_weight,
        swiglu_limit,
        n_threads,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// 6 专家并行 FFN（IQ2_XXS+Q2_K 混合量化）
#[pyfunction]
fn mixed_ffn_6experts_iq2xxs_q2k<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weights: &Bound<'py, PyList>,
    up_weights: &Bound<'py, PyList>,
    down_weights: &Bound<'py, PyList>,
    route_weights: Vec<f32>,
    swiglu_limit: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;

    let n_experts = gate_weights.len();

    // 收集 PyRef（保持 Python 对象存活）
    let gate_pys: Vec<PyRef<'_, Iq2XxsWeight>> = (0..n_experts)
        .map(|i| gate_weights.get_item(i)?.extract()).collect::<PyResult<_>>()?;
    let up_pys: Vec<PyRef<'_, Iq2XxsWeight>> = (0..n_experts)
        .map(|i| up_weights.get_item(i)?.extract()).collect::<PyResult<_>>()?;
    let down_pys: Vec<PyRef<'_, Q2KWeight>> = (0..n_experts)
        .map(|i| down_weights.get_item(i)?.extract()).collect::<PyResult<_>>()?;

    let gate_refs: Vec<&kernel::Iq2XxsWeight> = gate_pys.iter().map(|w| &w.inner).collect();
    let up_refs: Vec<&kernel::Iq2XxsWeight> = up_pys.iter().map(|w| &w.inner).collect();
    let down_refs: Vec<&kernel::Q2KWeight> = down_pys.iter().map(|w| &w.inner).collect();

    let result = kernel::mixed_ffn_6experts_iq2xxs_q2k(
        x_slice,
        &gate_refs,
        &up_refs,
        &down_refs,
        &route_weights,
        swiglu_limit,
    );
    Ok(PyArray1::from_vec(py, result))
}
#[pyfunction]
fn mixed_ffn_pair_iq2xxs_q2k_tiled<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;

    // 权重重排（一次性开销）
    let gate_tiled = kernel::Iq2XxsWeightTiled::from_weight(&gate_weight.inner);
    let up_tiled = kernel::Iq2XxsWeightTiled::from_weight(&up_weight.inner);
    let down_tiled = kernel::Q2KWeightTiled::from_weight(&down_weight.inner);

    let result = kernel::mixed_ffn_pair_iq2xxs_q2k_tiled(
        x_slice,
        &gate_tiled,
        &up_tiled,
        &down_tiled,
        route_weight,
        swiglu_limit,
    );
    Ok(PyArray1::from_vec(py, result))
}

/// CPU FFN 时间线分析（逐阶段计时）
#[pyfunction]
fn profile_mixed_ffn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
    n_iters: usize,
) -> PyResult<Vec<f64>> {
    use std::time::Instant;

    let x_slice = x.as_slice()?;
    let dim = gate_weight.inner.shape.1;
    let inter_dim = gate_weight.inner.shape.0;
    let n_blocks = dim / 256;
    let n_blocks_inter = inter_dim / 256;

    let mut timings = Vec::new();

    for _ in 0..n_iters {
        // Step 1: Q8 预量化
        let t0 = Instant::now();
        let mut q8_buf = vec![0i8; n_blocks * 256];
        let mut q8_inv_scales = vec![0.0f32; n_blocks];
        for blk in 0..n_blocks {
            q8_inv_scales[blk] = avx512::quantize_f32_to_q8_block(
                &x_slice[blk * 256..(blk + 1) * 256],
                &mut q8_buf[blk * 256..(blk + 1) * 256],
                256,
            );
        }
        timings.push(t0.elapsed().as_secs_f64() * 1000.0);

        // Step 2: gate 投影
        let t1 = Instant::now();
        let gate: Vec<f32> = {
            use rayon::prelude::*;
            use kernel::QuantizedWeight;
            (0..inter_dim).into_par_iter().map(|row| {
                unsafe { gate_weight.inner.vec_dot_q8(row, &q8_buf, &q8_inv_scales) }
            }).collect()
        };
        timings.push(t1.elapsed().as_secs_f64() * 1000.0);

        // Step 3: up 投影
        let t2 = Instant::now();
        let up: Vec<f32> = {
            use rayon::prelude::*;
            use kernel::QuantizedWeight;
            (0..inter_dim).into_par_iter().map(|row| {
                unsafe { up_weight.inner.vec_dot_q8(row, &q8_buf, &q8_inv_scales) }
            }).collect()
        };
        timings.push(t2.elapsed().as_secs_f64() * 1000.0);

        // Step 4: SwiGLU
        let t3 = Instant::now();
        let mut mid = vec![0.0f32; inter_dim];
        {
            use rayon::prelude::*;
            mid.par_iter_mut().enumerate().for_each(|(i, m)| {
                let g = gate[i];
                let mut u = up[i];
                if swiglu_limit > 0.0 { u = u.clamp(-swiglu_limit, swiglu_limit); }
                let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
                let sigmoid_g = 1.0 / (1.0 + kernel::exp_approx_scalar(-g_clamped));
                *m = g * sigmoid_g * u * route_weight;
            });
        }
        timings.push(t3.elapsed().as_secs_f64() * 1000.0);

        // Step 5: Q8 预量化 mid
        let t4 = Instant::now();
        let mut q8_buf2 = vec![0i8; n_blocks_inter * 256];
        let mut q8_inv_scales2 = vec![0.0f32; n_blocks_inter];
        for blk in 0..n_blocks_inter {
            q8_inv_scales2[blk] = avx512::quantize_f32_to_q8_block(
                &mid[blk * 256..(blk + 1) * 256],
                &mut q8_buf2[blk * 256..(blk + 1) * 256],
                256,
            );
        }
        timings.push(t4.elapsed().as_secs_f64() * 1000.0);

        // Step 6: down 投影
        let t5 = Instant::now();
        let _output: Vec<f32> = {
            use rayon::prelude::*;
            use kernel::QuantizedWeight;
            (0..down_weight.inner.n_rows()).into_par_iter().map(|row| {
                unsafe { down_weight.inner.vec_dot_q8(row, &q8_buf2, &q8_inv_scales2) }
            }).collect()
        };
        timings.push(t5.elapsed().as_secs_f64() * 1000.0);
    }

    Ok(timings)
}

/// SIMD 优化的路由计算（零拷贝 numpy 输入）。
#[pyfunction]
#[pyo3(signature = (scores, topk, bias=None, score_func="sqrtsoftplus", route_scale=1.0))]
fn route_simd<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f32>,
    topk: usize,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    score_func: &str,
    route_scale: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let scores_slice = scores.as_slice()?;
    let arr = scores.as_array();
    let n_experts = arr.shape()[1];
    let bias_slice = bias.as_ref().map(|b| b.as_slice()).transpose()?;

    let (weights, indices) = route::route_simd_impl(
        scores_slice,
        bias_slice,
        n_experts,
        topk,
        score_func,
        route_scale,
    );
    Ok((
        PyArray1::from_vec(py, weights),
        PyArray1::from_vec(py, indices),
    ))
}

/// 性能测试：单行 vec_dot 延迟
#[pyfunction]
fn bench_single_vec_dot<'py>(
    py: Python<'py>,
    weight: &Iq2XxsWeight,
    x: PyReadonlyArray1<'py, f32>,
    n_iters: usize,
) -> PyResult<f64> {
    use std::time::Instant;
    use kernel::QuantizedWeight;

    let x_slice = x.as_slice()?;
    let n_blocks = weight.inner.n_blocks_per_row();

    // 预量化
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = avx512::quantize_f32_to_q8_block(
            &x_slice[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    // Warmup
    for _ in 0..10 {
        unsafe { weight.inner.vec_dot_q8(0, &q8_buf, &q8_inv_scales) };
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..n_iters {
        unsafe { weight.inner.vec_dot_q8(0, &q8_buf, &q8_inv_scales) };
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(elapsed / n_iters as f64 * 1000.0) // ms
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_avx512_supported, m)?)?;
    m.add_function(wrap_pyfunction!(is_avx2_supported, m)?)?;
    m.add_function(wrap_pyfunction!(init_tables, m)?)?;
    m.add_function(wrap_pyfunction!(is_tables_initialized, m)?)?;
    m.add_class::<Iq2XsWeight>()?;
    m.add_class::<Iq2XxsWeight>()?;
    m.add_class::<Q2KWeight>()?;
    m.add_class::<Fp4Weight>()?;
    m.add_class::<Iq2XsTile>()?;
    m.add_function(wrap_pyfunction!(cpu_expert_ffn_pair, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_expert_ffn_pair_fp4, m)?)?;
    m.add_function(wrap_pyfunction!(mixed_ffn_pair_iq2xxs_q2k, m)?)?;
    m.add_function(wrap_pyfunction!(mixed_ffn_pair_iq2xxs_q2k_streaming, m)?)?;
    m.add_function(wrap_pyfunction!(mixed_ffn_pair_iq2xxs_q2k_blocked_mt, m)?)?;
    m.add_function(wrap_pyfunction!(mixed_ffn_6experts_iq2xxs_q2k, m)?)?;
    m.add_function(wrap_pyfunction!(mixed_ffn_pair_iq2xxs_q2k_tiled, m)?)?;
    m.add_function(wrap_pyfunction!(profile_mixed_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_expert_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_expert_ffn_pair_dual, m)?)?;
    m.add_function(wrap_pyfunction!(iq2xs_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_expert_ffn_pair_tile, m)?)?;
    m.add_function(wrap_pyfunction!(iq2xs_matvec_tile, m)?)?;
    m.add_function(wrap_pyfunction!(route_simd, m)?)?;
    m.add_function(wrap_pyfunction!(bench_single_vec_dot, m)?)?;
    m.add_class::<CpuExpertRunner>()?;
    m.add_class::<CpuFfnEngineIq2xxsQ2k>()?;
    Ok(())
}
