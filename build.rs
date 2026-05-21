// build.rs — 编译 csrc/ 中的 C 代码
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let csrc_dir = PathBuf::from(&manifest_dir).join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 手动编译 C 代码为共享库
    let bridge_src = csrc_dir.join("iq2_xs_bridge.c");
    let so_path = out_dir.join("libiq2_xs_c.so");

    // 直接编译为共享库
    let status = std::process::Command::new("cc")
        .args([
            "-O3", "-mavx2", "-mfma", "-fPIC", "-shared",
            "-I", csrc_dir.to_str().unwrap(),
            bridge_src.to_str().unwrap(),
            "-o", so_path.to_str().unwrap(),
            "-lm",
        ])
        .status()
        .expect("cc 编译失败");
    assert!(status.success(), "C 编译/链接失败");

    // 验证符号
    let output = std::process::Command::new("nm")
        .arg("-D")
        .arg(&so_path)
        .output()
        .expect("nm 失败");
    let symbols = String::from_utf8_lossy(&output.stdout);
    assert!(symbols.contains("iq2xs_init_ffi"), "FFI 符号未找到");

    // 告诉 cargo 链接
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=iq2_xs_c");

    // 重新编译触发条件
    println!("cargo:rerun-if-changed=csrc/iq2_xs_bridge.c");
    println!("cargo:rerun-if-changed=csrc/iq2_xs.h");
}
