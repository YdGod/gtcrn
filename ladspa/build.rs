// =============================================================================
// GTCRN LADSPA 插件 Rust 构建脚本 (build.rs)
// =============================================================================
// 职责：
// 1. 静态构建时：发现并链接精简版 ONNX Runtime 的静态库
// 2. 将 ONNX 模型转换为 ORT 格式（优化后的 ONNX Runtime 专用格式）
//
// 注意：
// - 此脚本在 cargo build 之前运行
// - 链接静态库时，需正确处理 Abseil、protobuf 等依赖的顺序
// =============================================================================
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // 默认启用模型格式转换（ONNX → ORT 优化格式）
    convert_models(&manifest_dir);

    // 非静态构建 → 跳过静态链接配置
    if !cfg!(feature = "static") {
        return;
    }

    // ================= 静态链接：查找精简版 ONNX Runtime 库 =================
    let lib_dir = PathBuf::from(&manifest_dir)
        .join("onnxruntime-minimal")  // Docker 构建输出的目录
        .join("lib");

    if !lib_dir.exists() {
        panic!(
            "静态 ONNX Runtime 库未找到于 {:?}。请先运行 ./build-minimal-docker.sh。",
            lib_dir
        );
    }

    // 告诉 rustc 在此目录搜索库文件
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // 链接统一的 libonnxruntime.a（由 build.sh 中的 MRI 脚本合并所有 .a 文件）
    // +whole-archive 确保 OrtGetApiBase 等符号被包含（即使 Rust 代码未直接引用）
    println!("cargo:rustc-link-lib=static:+whole-archive=onnxruntime");

    // ================= 自动发现并链接 Abseil 库 =================
    // Abseil 是 Google 的 C++ 基础库，ONNX Runtime 依赖它
    // 按字母顺序排序以保持确定性构建
    let mut absl_libs = Vec::new();
    if let Ok(entries) = fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap();
            if name.starts_with("libabsl_") && name.ends_with(".a") {
                let lib_name = &name[3..name.len() - 2]; // 去掉 "lib" 前缀和 ".a" 后缀
                absl_libs.push(lib_name.to_string());
            }
        }
    }
    absl_libs.sort();

    for lib in absl_libs {
        println!("cargo:rustc-link-lib=static={}", lib);
    }

    // ================= 链接其他必需的 C/C++ 依赖库 =================
    let deps = [
        "protobuf",        // Protocol Buffers（模型序列化）
        "protobuf-lite",   // protobuf 精简版
        "nsync_cpp",       // Google 的同步库
        "cpuinfo",         // CPU 特性检测
        "flatbuffers",     // FlatBuffers 序列化（ORT 内部使用）
    ];
    for dep in deps {
        let filename = format!("lib{}.a", dep);
        if lib_dir.join(&filename).exists() {
            println!("cargo:rustc-link-lib=static={}", dep);
        }
    }

    // ================= 链接系统运行时库 =================
    println!("cargo:rustc-link-lib=pthread");   // POSIX 线程
    println!("cargo:rustc-link-lib=dl");        // 动态链接库加载
    println!("cargo:rustc-link-lib=stdc++");    // C++ 标准库（ORT 是 C++ 编写的）
}

/// 将 ONNX 模型转换为 ORT 优化格式
///
/// ORT 格式是 ONNX Runtime 的专有格式，相比原始 ONNX：
/// - 消除了不必要的 Range 等算子
/// - 预分配了内存布局
/// - 减少了首次加载时间
fn convert_models(manifest_dir: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 定义需要转换的模型：(源文件名, 输出文件名)
    let models = [
        ("gtcrn.onnx", "gtcrn.ort"),            // 完整质量模型
        ("gtcrn_simple.onnx", "gtcrn_simple.ort"), // 轻量简化模型
    ];

    // 模型文件路径
    let stream_dir = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("stream")
        .join("onnx_models");

    // Python 虚拟环境路径（用于运行 onnxruntime 转换工具）
    let python_path = PathBuf::from(manifest_dir).join(".venv/bin/python");

    if !python_path.exists() {
        println!("cargo:warning=Python 虚拟环境未找到于 {:?}, 跳过模型转换。如需转换模型，请创建 .venv。", python_path);
        return;
    }

    for (src_name, dst_name) in models {
        let src_path = stream_dir.join(src_name);
        let dst_path = out_dir.join(dst_name);

        // 当源文件变化时重新运行此脚本
        println!("cargo:rerun-if-changed={}", src_path.display());

        if !src_path.exists() {
            println!(
                "cargo:warning=源模型文件未找到于 {:?}, 跳过转换。",
                src_path
            );
            continue;
        }

        // 判断是否需要重新转换（目标不存在 或 源文件更新）
        let should_convert = if !dst_path.exists() {
            true
        } else {
            let src_meta = fs::metadata(&src_path).unwrap();
            let dst_meta = fs::metadata(&dst_path).unwrap();
            src_meta.modified().unwrap() > dst_meta.modified().unwrap()
        };

        if should_convert {
            println!("正在将 {} 转换为 ORT 格式...", src_name);
            let status = std::process::Command::new(&python_path)
                .args([
                    "-m",
                    "onnxruntime.tools.convert_onnx_models_to_ort",
                    src_path.to_str().unwrap(),
                    "--output_dir",
                    out_dir.to_str().unwrap(),
                    "--optimization_style",
                    "Fixed",  // Fixed = 固定优化（适用于已预优化的模型）
                              // Runtime = 运行时优化（首次加载时优化）
                ])
                .status()
                .expect("模型转换命令执行失败");

            if !status.success() {
                panic!("模型转换失败: {}", src_name);
            }
        }
    }
}
