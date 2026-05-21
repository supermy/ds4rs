use anyhow::{anyhow, Result};

pub struct PinnedBuffer {
    ptr: *mut u8,
    len: usize,
}

impl PinnedBuffer {
    pub fn alloc(len: usize) -> Result<Self> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, len) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemAllocHost failed: {:?}", result));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    pub fn copy_from(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.len {
            return Err(anyhow!(
                "pinned buffer too small: {} < {}",
                self.len,
                data.len()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        Ok(())
    }

    pub fn host_ptr(&self) -> u64 {
        self.ptr as u64
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

pub struct PinnedPool {
    default_size: usize,
}

impl PinnedPool {
    pub fn new(default_size: usize) -> Self {
        Self { default_size }
    }

    pub fn get(&mut self, min_size: usize) -> Result<PinnedBuffer> {
        let size = min_size.max(self.default_size);
        PinnedBuffer::alloc(size)
    }
}
