use crate::dtype::DType;
use std::ffi::c_void;

#[repr(i32)]
pub enum DLDeviceType {
    CPU = 1,
    CUDA = 2,
    CUDAHost = 3,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

impl DLDevice {
    pub fn cuda(device_id: i32) -> Self {
        Self {
            device_type: DLDeviceType::CUDA as i32,
            device_id,
        }
    }

    pub fn cpu() -> Self {
        Self {
            device_type: DLDeviceType::CPU as i32,
            device_id: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl DLDataType {
    pub fn from_dtype(dtype: DType) -> Self {
        let (code, bits, lanes) = dtype.dlpack_type_code();
        Self { code, bits, lanes }
    }

    pub fn new(code: u8, bits: u8, lanes: u16) -> Self {
        Self { code, bits, lanes }
    }
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

impl DLTensor {
    pub fn new(
        data_ptr: u64,
        shape: &[usize],
        dtype: DType,
        device: DLDevice,
    ) -> Self {
        let ndim = shape.len() as i32;
        let dl_dtype = DLDataType::from_dtype(dtype);
        let shape_ptr = shape.as_ptr() as *mut i64;
        Self {
            data: data_ptr as *mut c_void,
            device,
            ndim,
            dtype: dl_dtype,
            shape: shape_ptr,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        }
    }
}

unsafe impl Send for DLTensor {}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

unsafe impl Send for DLManagedTensor {}
