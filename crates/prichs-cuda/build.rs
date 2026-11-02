use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_msvc_cl() -> Option<PathBuf> {
    // If cl.exe is already runnable in PATH, no need to inject
    if Command::new("cl").output().is_ok() {
        return None;
    }

    let editions = ["Community", "Professional", "Enterprise", "BuildTools"];
    for year in ["2022", "2019"] {
        for ed in editions {
            let base = PathBuf::from(format!(
                r"C:\Program Files\Microsoft Visual Studio\{}\{}\VC\Tools\MSVC",
                year, ed
            ));
            if base.exists() {
                if let Ok(entries) = std::fs::read_dir(&base) {
                    for entry in entries.flatten() {
                        let candidate = entry.path().join(r"bin\Hostx64\x64");
                        if candidate.join("cl.exe").exists() {
                            return Some(candidate);
                        }
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let is_windows = cfg!(target_os = "windows");

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
        if is_windows {
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1".to_string()
        } else {
            "/usr/local/cuda".to_string()
        }
    });

    if is_windows {
        println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=static=kernels");
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let temp_build = env::temp_dir().join("prichs_cuda_build");
    let _ = std::fs::create_dir_all(&temp_build);

    let temp_cu = temp_build.join("kernels.cu");
    let lib_filename = if is_windows { "kernels.lib" } else { "libkernels.a" };
    let temp_lib = temp_build.join(lib_filename);

    let source_cu = Path::new(&manifest_dir).join("src").join("kernels.cu");
    let _ = std::fs::copy(&source_cu, &temp_cu);

    let mut nvcc_cmd = Command::new("nvcc");

    if is_windows {
        if let Some(msvc_bin) = find_msvc_cl() {
            if let Ok(current_path) = env::var("PATH") {
                let new_path = format!("{};{}", msvc_bin.display(), current_path);
                nvcc_cmd.env("PATH", new_path);
            }
        }
    }

    let mut cmd = nvcc_cmd;
    cmd.arg("-O3")
        .arg("-lib")
        .arg(&temp_cu)
        .arg("-o")
        .arg(&temp_lib);

    if is_windows {
        cmd.arg("-allow-unsupported-compiler")
            .arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
    } else {
        cmd.arg("-Xcompiler").arg("-fPIC");
    }

    let output = cmd.output();

    match output {
        Ok(out) if out.status.success() => {
            let dest_lib = Path::new(&out_dir).join(lib_filename);
            std::fs::copy(&temp_lib, &dest_lib).expect("Failed to copy kernels library to OUT_DIR");
            println!("cargo:warning=CUDA kernels compiled and copied to OUT_DIR");
        }
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            panic!("nvcc failed: stdout={}\nstderr={}", stdout, stderr);
        }
        Err(e) => {
            panic!("Failed to spawn nvcc (is CUDA toolkit installed and in PATH?): {:?}", e);
        }
    }
}
