use crate::tensor::GpuTensor;
use anyhow::{anyhow, Result};
use cudarc::cublas::result;
use cudarc::cublas::sys;
use cudarc::driver::CudaContext;
use std::sync::Arc;

pub struct CublasHandle {
    handle: sys::cublasHandle_t,
}

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    pub fn new(device: Arc<CudaContext>) -> Result<Self> {
        let handle = result::create_handle().map_err(|e| anyhow!("cublasCreate failed: {:?}", e))?;
        let stream = device.default_stream();
        {
            let cu_stream = stream.cu_stream() as *mut sys::CUstream_st;
            unsafe {
                result::set_stream(handle, cu_stream)
                    .map_err(|e| anyhow!("cublasSetStream failed: {:?}", e))?;
            }
        }
        Ok(Self { handle })
    }

    pub fn set_stream(&self, stream: sys::cudaStream_t) -> Result<()> {
        unsafe {
            result::set_stream(self.handle, stream)
                .map_err(|e| anyhow!("cublasSetStream failed: {:?}", e))
        }
    }

    pub fn gemm_bf16(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr();
        let b_ptr = b.device_ptr();
        let c_ptr = c.device_ptr();

        unsafe {
            result::gemm_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS BF16 GEMM failed: {:?}", e))?
        }
        Ok(())
    }

    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr();
        let b_ptr = b.device_ptr();
        let c_ptr = c.device_ptr();

        unsafe {
            result::gemm_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS FP32 GEMM failed: {:?}", e))?
        }
        Ok(())
    }

    pub fn gemm_bf16_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_size: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr();
        let b_ptr = b.device_ptr();
        let c_ptr = c.device_ptr();

        unsafe {
            result::gemm_strided_batched_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                stride_b,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                stride_a,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                stride_c,
                batch_size,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS BF16 strided batched GEMM failed: {:?}", e))?
        }
        Ok(())
    }

    pub fn gemm_bf16_nn_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_size: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr();
        let b_ptr = b.device_ptr();
        let c_ptr = c.device_ptr();

        unsafe {
            result::gemm_strided_batched_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                stride_b,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                stride_a,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                stride_c,
                batch_size,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS BF16 NN strided batched GEMM failed: {:?}", e))?
        }
        Ok(())
    }

    pub fn gemm_bf16_tn(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        lda: i32,
        a_elem_offset: usize,
        b: &GpuTensor,
        ldb: i32,
        b_elem_offset: usize,
        c: &mut GpuTensor,
        ldc: i32,
        c_elem_offset: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr() + (a_elem_offset * 2) as u64;
        let b_ptr = b.device_ptr() + (b_elem_offset * 2) as u64;
        let c_ptr = c.device_ptr() + (c_elem_offset * 2) as u64;

        unsafe {
            result::gemm_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                m as i32,
                n as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                lda,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                ldb,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                ldc,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS BF16 TN GEMM failed: {:?}", e))?
        }
        Ok(())
    }

    pub fn gemm_f32_nn_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_size: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr();
        let b_ptr = b.device_ptr();
        let c_ptr = c.device_ptr();

        unsafe {
            result::gemm_strided_batched_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_b,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_a,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch_size,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS FP32 NN strided batched GEMM failed: {:?}", e))?
        }
        Ok(())
    }

    /// 行主序 C = A^T @ B 的分批步进 GEMM
    /// 利用 cuBLAS 列主序技巧：C_col = B_col @ A_col^T
    /// 即 A_cublas=B_row(opN), B_cublas=A_row(opT)
    pub fn gemm_f32_tn_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_size: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a_ptr = a.device_ptr();
        let b_ptr = b.device_ptr();
        let c_ptr = c.device_ptr();

        unsafe {
            result::gemm_strided_batched_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_T,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32) as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_b,
                a_ptr as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_a,
                (&beta as *const f32) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch_size,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| anyhow!("cuBLAS FP32 TN strided batched GEMM failed: {:?}", e))?
        }
        Ok(())
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = result::destroy_handle(self.handle);
        }
    }
}
