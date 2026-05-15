use crate::dlpack::{DLDevice, DLTensor};
use crate::tensor::GpuTensor;
use anyhow::{anyhow, Context, Result};
use libloading::os::unix::Library as UnixLibrary;
use libloading::Library;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const K_TVMFFI_NONE: i32 = 0;
const K_TVMFFI_BOOL: i32 = 2;
const K_TVMFFI_RAW_STR: i32 = 8;
const K_TVMFFI_DL_TENSOR_PTR: i32 = 7;
const K_TVMFFI_FUNCTION: i32 = 68;
const K_TVMFFI_MODULE: i32 = 73;

#[repr(C)]
struct TVMFFIAny {
    type_index: i32,
    zero_padding: u32,
    v_int64: i64,
}

impl TVMFFIAny {
    fn none() -> Self {
        Self { type_index: K_TVMFFI_NONE, zero_padding: 0, v_int64: 0 }
    }

    fn from_bool(v: bool) -> Self {
        Self { type_index: K_TVMFFI_BOOL, zero_padding: 0, v_int64: if v { 1 } else { 0 } }
    }

    fn from_raw_str(s: *const u8) -> Self {
        Self { type_index: K_TVMFFI_RAW_STR, zero_padding: 0, v_int64: s as i64 }
    }

    fn from_object(type_index: i32, handle: i64) -> Self {
        Self { type_index, zero_padding: 0, v_int64: handle }
    }

    fn from_dl_tensor_ptr(ptr: *const DLTensor) -> Self {
        Self { type_index: K_TVMFFI_DL_TENSOR_PTR, zero_padding: 0, v_int64: ptr as i64 }
    }
}

#[repr(C)]
struct TVMFFIByteArray {
    data: *const u8,
    size: usize,
}

type FnGetGlobal = unsafe extern "C" fn(*const TVMFFIByteArray, *mut *mut c_void) -> i32;
type FnFunctionCall = unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;
type FnObjectIncRef = unsafe extern "C" fn(*mut c_void) -> i32;
type FnObjectDecRef = unsafe extern "C" fn(*mut c_void) -> i32;
type FnErrorMoveFromRaised = unsafe extern "C" fn(*mut *mut c_void);

struct TvmCApi {
    get_global: FnGetGlobal,
    function_call: FnFunctionCall,
    object_inc_ref: FnObjectIncRef,
    object_dec_ref: FnObjectDecRef,
    error_move_from_raised: FnErrorMoveFromRaised,
    _deps: Vec<Library>,
    _lib: Library,
}

impl TvmCApi {
    fn load(lib_path: &Path, extra_deps: Vec<PathBuf>) -> Result<Self> {
        unsafe {
            for dep in &extra_deps {
                if dep.exists() {
                    match UnixLibrary::open(Some(dep), libc::RTLD_NOW | libc::RTLD_GLOBAL) {
                        Ok(_lib) => {}
                        Err(e) => eprintln!("warning: failed to load dep {}: {}", dep.display(), e),
                    }
                }
            }

            let lib = Library::new(lib_path)
                .with_context(|| format!("failed to load {}", lib_path.display()))?;

            let get_global: FnGetGlobal = *lib
                .get(b"TVMFFIFunctionGetGlobal\0")
                .with_context(|| "TVMFFIFunctionGetGlobal not found")?;

            let function_call: FnFunctionCall = *lib
                .get(b"TVMFFIFunctionCall\0")
                .with_context(|| "TVMFFIFunctionCall not found")?;

            let object_inc_ref: FnObjectIncRef = *lib
                .get(b"TVMFFIObjectIncRef\0")
                .with_context(|| "TVMFFIObjectIncRef not found")?;

            let object_dec_ref: FnObjectDecRef = *lib
                .get(b"TVMFFIObjectDecRef\0")
                .with_context(|| "TVMFFIObjectDecRef not found")?;

            let error_move_from_raised: FnErrorMoveFromRaised = *lib
                .get(b"TVMFFIErrorMoveFromRaised\0")
                .with_context(|| "TVMFFIErrorMoveFromRaised not found")?;

            Ok(Self {
                get_global,
                function_call,
                object_inc_ref,
                object_dec_ref,
                error_move_from_raised,
                _deps: Vec::new(),
                _lib: lib,
            })
        }
    }

    unsafe fn get_global_func(&self, name: &[u8]) -> Result<*mut c_void> {
        let name_with_nul = format!("{}\0", std::str::from_utf8(name).unwrap_or(""));
        let ba = TVMFFIByteArray {
            data: name_with_nul.as_ptr(),
            size: name.len(),
        };
        let mut handle: *mut c_void = std::ptr::null_mut();
        let ret = (self.get_global)(&ba, &mut handle);
        if ret != 0 {
            return Err(anyhow!("TVMFFIFunctionGetGlobal({:?}) failed: {}", name, ret));
        }
        if handle.is_null() {
            return Err(anyhow!("global function {:?} not found", name));
        }
        Ok(handle)
    }

    unsafe fn call_function(
        &self,
        func: *mut c_void,
        args: &[TVMFFIAny],
    ) -> Result<TVMFFIAny> {
        let mut result = TVMFFIAny::none();
        let ret = (self.function_call)(func, args.as_ptr(), args.len() as i32, &mut result);
        if ret != 0 {
            let mut err_handle: *mut c_void = std::ptr::null_mut();
            (self.error_move_from_raised)(&mut err_handle);
            return Err(anyhow!(
                "TVMFFIFunctionCall failed (ret={}, err_handle={:?})",
                ret,
                err_handle
            ));
        }
        Ok(result)
    }

    unsafe fn inc_ref(&self, handle: *mut c_void) {
        (self.object_inc_ref)(handle);
    }

    unsafe fn dec_ref(&self, handle: *mut c_void) {
        (self.object_dec_ref)(handle);
    }
}

fn find_tilelang_lib_dir() -> Option<PathBuf> {
    if let Ok(out) = std::process::Command::new("python3")
        .args(["-c", "import tilelang, os; print(os.path.join(os.path.dirname(tilelang.__file__), 'lib'))"])
        .output()
    {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if Path::new(&p).exists() {
                return Some(PathBuf::from(p));
            }
        }
    }
    let candidates = [
        "/usr/local/lib/python3.12/dist-packages/tilelang/lib",
        "/usr/local/lib/python3.11/dist-packages/tilelang/lib",
        "/usr/local/lib/python3.10/dist-packages/tilelang/lib",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return Some(PathBuf::from(c));
        }
    }
    None
}

fn find_libtvm_ffi() -> Result<PathBuf> {
    let candidates = [
        "/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib/libtvm_ffi.so",
        "/usr/local/lib/python3.11/dist-packages/tvm_ffi/lib/libtvm_ffi.so",
        "/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib/libtvm_ffi.so",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return Ok(PathBuf::from(c));
        }
    }
    if let Ok(out) = std::process::Command::new("python3")
        .args(["-c", "import tvm_ffi; print(tvm_ffi.LIB._name)"])
        .output()
    {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if Path::new(&p).exists() {
                return Ok(PathBuf::from(p));
            }
        }
    }
    Err(anyhow!("libtvm_ffi.so not found. Install with: pip install apache-tvm-ffi"))
}

pub struct TvmRuntime {
    api: Arc<TvmCApi>,
    lib_path: PathBuf,
}

impl TvmRuntime {
    pub fn new() -> Result<Self> {
        let lib_path = find_libtvm_ffi()?;

        let mut deps = Vec::new();
        if let Some(tl_dir) = find_tilelang_lib_dir() {
            deps.push(tl_dir.join("libtvm.so"));
            deps.push(tl_dir.join("libtilelang.so"));
        }

        let api = TvmCApi::load(&lib_path, deps)?;
        Ok(Self { api: Arc::new(api), lib_path })
    }

    pub fn with_lib_path(path: &str) -> Result<Self> {
        let lib_path = PathBuf::from(path);
        let mut deps = Vec::new();
        if let Some(tl_dir) = find_tilelang_lib_dir() {
            deps.push(tl_dir.join("libtvm.so"));
            deps.push(tl_dir.join("libtilelang.so"));
        }
        let api = TvmCApi::load(&lib_path, deps)?;
        Ok(Self { api: Arc::new(api), lib_path })
    }

    pub fn lib_path(&self) -> &Path {
        &self.lib_path
    }
}

struct TvmModuleHandle {
    handle: *mut c_void,
    api: Arc<TvmCApi>,
}

impl Drop for TvmModuleHandle {
    fn drop(&mut self) {
        unsafe { self.api.dec_ref(self.handle); }
    }
}

struct TvmFuncHandle {
    handle: *mut c_void,
    api: Arc<TvmCApi>,
}

impl Drop for TvmFuncHandle {
    fn drop(&mut self) {
        unsafe { self.api.dec_ref(self.handle); }
    }
}

pub struct TlKernel {
    _module: TvmModuleHandle,
    func: TvmFuncHandle,
    api: Arc<TvmCApi>,
}

impl TlKernel {
    pub fn load(runtime: &TvmRuntime, so_path: &str, func_name: &str) -> Result<Self> {
        unsafe {
            let api = &runtime.api;

            let load_func = api.get_global_func(b"ffi.ModuleLoadFromFile")?;

            let path_cstr = format!("{}\0", so_path);
            let arg_path = TVMFFIAny::from_raw_str(path_cstr.as_ptr());

            let result = api.call_function(load_func, &[arg_path])?;
            if result.type_index != K_TVMFFI_MODULE {
                return Err(anyhow!(
                    "ModuleLoadFromFile returned type_index={}, expected {}",
                    result.type_index,
                    K_TVMFFI_MODULE
                ));
            }
            let module_handle = result.v_int64 as *mut c_void;
            api.inc_ref(module_handle);

            let get_func = api.get_global_func(b"ffi.ModuleGetFunction")?;

            let arg_mod = TVMFFIAny::from_object(K_TVMFFI_MODULE, module_handle as i64);
            let fname_cstr = format!("{}\0", func_name);
            let arg_fname = TVMFFIAny::from_raw_str(fname_cstr.as_ptr());
            let arg_query = TVMFFIAny::from_bool(false);

            let result2 = api.call_function(get_func, &[arg_mod, arg_fname, arg_query])?;
            if result2.type_index != K_TVMFFI_FUNCTION {
                return Err(anyhow!(
                    "ModuleGetFunction returned type_index={}, expected {}",
                    result2.type_index,
                    K_TVMFFI_FUNCTION
                ));
            }
            let func_handle = result2.v_int64 as *mut c_void;
            api.inc_ref(func_handle);

            Ok(Self {
                _module: TvmModuleHandle {
                    handle: module_handle,
                    api: Arc::clone(api),
                },
                func: TvmFuncHandle {
                    handle: func_handle,
                    api: Arc::clone(api),
                },
                api: Arc::clone(api),
            })
        }
    }

    pub fn call(&self, tensors: &[&GpuTensor]) -> Result<()> {
        unsafe {
            let mut dl_tensors: Vec<DLTensor> = Vec::with_capacity(tensors.len());

            for tensor in tensors {
                let dl = DLTensor::new(
                    tensor.device_ptr(),
                    tensor.shape.as_slice(),
                    tensor.dtype,
                    DLDevice::cuda(0),
                );
                dl_tensors.push(dl);
            }

            let args: Vec<TVMFFIAny> = dl_tensors
                .iter()
                .map(|dl| TVMFFIAny::from_dl_tensor_ptr(dl))
                .collect();

            self.api.call_function(self.func.handle, &args)?;
            Ok(())
        }
    }
}

pub struct KernelRegistry {
    runtime: Arc<TvmRuntime>,
    kernels: Mutex<HashMap<String, TlKernel>>,
}

impl KernelRegistry {
    pub fn new(runtime: Arc<TvmRuntime>) -> Self {
        Self { runtime, kernels: Mutex::new(HashMap::new()) }
    }

    pub fn load(&self, so_path: &str, func_name: &str) -> Result<()> {
        let kernel = TlKernel::load(&self.runtime, so_path, func_name)?;
        let key = Path::new(so_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("{}/{}", so_path, func_name));
        self.kernels.lock().unwrap().insert(key, kernel);
        Ok(())
    }

    pub fn call(&self, name: &str, tensors: &[&GpuTensor]) -> Result<()> {
        let kernels = self.kernels.lock().unwrap();
        let kernel = kernels
            .get(name)
            .ok_or_else(|| anyhow!("kernel {} not found in registry", name))?;
        kernel.call(tensors)
    }

    pub fn load_dir(&self, dir: &str) -> Result<usize> {
        let dir_path = Path::new(dir);
        if !dir_path.exists() {
            return Err(anyhow!("kernel directory {} not found", dir));
        }

        let manifest_path = dir_path.join("kernels.json");
        if manifest_path.exists() {
            return self.load_from_manifest(dir, &manifest_path);
        }

        let mut count = 0;
        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "so") {
                let filename = path.file_name().unwrap().to_string_lossy().to_string();
                let kernel_name = filename.trim_end_matches(".so").to_string();
                let func_name = kernel_name.clone() + "_";
                let so_path = path.to_string_lossy().to_string();
                match TlKernel::load(&self.runtime, &so_path, &func_name) {
                    Ok(kernel) => {
                        self.kernels.lock().unwrap().insert(kernel_name, kernel);
                        count += 1;
                    }
                    Err(e) => { eprintln!("warning: failed to load kernel {}: {}", so_path, e); }
                }
            }
        }
        Ok(count)
    }

    fn load_from_manifest(&self, dir: &str, manifest_path: &Path) -> Result<usize> {
        let content = std::fs::read_to_string(manifest_path)
            .with_context(|| format!("failed to read manifest {:?}", manifest_path))?;
        let manifest: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| "failed to parse kernels.json")?;
        let entries = manifest.as_object()
            .ok_or_else(|| anyhow!("manifest is not a JSON object"))?;

        let mut count = 0;
        for (name, info) in entries {
            let so_file = info.get("so").and_then(|v| v.as_str()).unwrap_or("");
            let func_name = info.get("func").and_then(|v| v.as_str()).unwrap_or("");
            if so_file.is_empty() || func_name.is_empty() {
                continue;
            }
            let so_path = format!("{}/{}", dir, so_file);
            match TlKernel::load(&self.runtime, &so_path, func_name) {
                Ok(kernel) => {
                    self.kernels.lock().unwrap().insert(name.clone(), kernel);
                    count += 1;
                }
                Err(e) => { eprintln!("warning: failed to load kernel {}: {}", so_path, e); }
            }
        }
        Ok(count)
    }
}

pub fn init_tvm_runtime() -> Result<Arc<TvmRuntime>> {
    Ok(Arc::new(TvmRuntime::new()?))
}
