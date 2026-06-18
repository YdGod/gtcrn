#!/bin/bash
# =============================================================================
# GTCRN LADSPA ORT 构建脚本
# =============================================================================
# 支持三种构建策略：
#   1. dynamic  — 动态链接系统安装的 ONNX Runtime
#   2. static   — 下载微软预编译的 ONNX Runtime（全量约50MB）
#   3. minimal  — 使用 Docker 自编译的精简 ONNX Runtime（约6MB）
#
# 用法: ./build.sh [dynamic|static|minimal]
# 默认: dynamic
# =============================================================================

set -e  # 遇到任何错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 终端颜色（用于美化输出）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # 无色（重置）

print_usage() {
    echo "用法: $0 [dynamic|static|minimal]"
    echo ""
    echo "选项说明:"
    echo "  dynamic  - 使用系统安装的 ONNX Runtime (libonnxruntime.so)"
    echo "             编译快，但需要系统已安装 'onnxruntime' 包"
    echo ""
    echo "  static   - 使用微软下载的预编译 ONNX Runtime（捆绑式）"
    echo "             下载完整版 Runtime (~50MB) 并内嵌到插件中"
    echo "             注意: 这里的 'static' 指捆绑依赖，实际仍是动态链接 .so"
    echo ""
    echo "  minimal  - 使用 Docker 编译的精简 ONNX Runtime（完全静态链接）"
    echo "             需要先运行 ./build-minimal-docker.sh"
    echo "             生成单个、小型、无外部依赖的 .so 插件文件"
    echo ""
    echo "默认: dynamic"
}

# ==================== 动态构建 ====================
build_dynamic() {
    echo -e "${GREEN}正在构建 DYNAMIC（系统 ORT 动态链接）...${NC}"
    echo "要求系统已安装 'onnxruntime' 包"
    echo ""

    # 使用系统CPU核心数并行编译
    CARGO_BUILD_JOBS=$(nproc)
    export CARGO_BUILD_JOBS
    cargo build --release --features dynamic --no-default-features

    BINARY="target/release/libgtcrn_ladspa_ort.so"
    if [ -f "$BINARY" ]; then
        SIZE=$(du -h "$BINARY" | cut -f1)
        echo ""
        echo -e "${GREEN}✓ 动态构建成功！${NC}"
        echo "  二进制文件: $BINARY"
        echo "  文件大小: $SIZE"
    fi
}

# ==================== 静态构建（微软预编译）====================
build_static() {
    echo -e "${YELLOW}正在构建 STATIC（微软下载预编译ORT）...${NC}"
    echo "从微软服务器下载预编译的 ONNX Runtime 并捆绑"
    echo ""

    CARGO_BUILD_JOBS=$(nproc)
    export CARGO_BUILD_JOBS
    # 'download' 特性启用 ort/download-binaries → 自动下载
    cargo build --release --features download --no-default-features

    BINARY="target/release/libgtcrn_ladspa_ort.so"
    if [ -f "$BINARY" ]; then
        SIZE=$(du -h "$BINARY" | cut -f1)
        echo ""
        echo -e "${GREEN}✓ 静态（下载）构建成功！${NC}"
        echo "  二进制文件: $BINARY"
        echo "  文件大小: $SIZE"
        echo "  注意: libonnxruntime.so 由构建过程自动下载/捆绑"
    fi
}

# ==================== 最小化构建（Docker自编译）====================
build_minimal() {
    echo -e "${YELLOW}正在构建 MINIMAL（Docker 精简ORT）...${NC}"
    echo "将精简版 ONNX Runtime 静态链接到插件中"

    # 检查 Docker 构建的输出目录是否存在
    if [ ! -d "onnxruntime-minimal/lib" ]; then
        echo -e "${RED}错误: onnxruntime-minimal/lib 目录未找到${NC}"
        echo "请先运行 ./build-minimal-docker.sh"
        exit 1
    fi

    # ======== 创建统一的 libonnxruntime.a ========
    # ORT 编译后会产生多个 .a 文件（onnxruntime, protobuf, absl等）
    # 需要合并为一个文件以便链接
    echo "正在创建统一的 libonnxruntime.a..."
    cd onnxruntime-minimal/lib

    # 使用 MRI (Master Record Index) 脚本合并多个静态库
    echo "CREATE libonnxruntime.a" >lib_script.mri
    for lib in libonnxruntime_*.a libonnx.a libonnx_proto.a; do
        if [ -f "$lib" ]; then
            echo "ADDLIB $lib" >>lib_script.mri
        fi
    done
    echo "SAVE" >>lib_script.mri
    echo "END" >>lib_script.mri

    ar -M <lib_script.mri     # 使用 ar 归档工具执行合并
    rm lib_script.mri         # 清理脚本
    cd ../..

    echo "正在嵌入 ONNX Runtime（静态链接）..."

    # 设置环境变量，告知 ort-sys 使用本地库而非系统库
    ORT_STRATEGY=system
    export ORT_STRATEGY
    ORT_LIB_LOCATION=$(pwd)/onnxruntime-minimal/lib
    export ORT_LIB_LOCATION
    CARGO_BUILD_JOBS=$(nproc)
    export CARGO_BUILD_JOBS

    # 'static' 特性启用 build.rs 中的静态链接逻辑
    cargo build --release --features static --no-default-features

    if [ -f "target/release/libgtcrn_ladspa_ort.so" ]; then
        SIZE=$(du -h target/release/libgtcrn_ladspa_ort.so | cut -f1)
        echo ""
        echo -e "${GREEN}✓ 最小化构建成功！${NC}"
        echo "  插件: target/release/libgtcrn_ladspa_ort.so ($SIZE)"
        echo "  依赖: 单文件（完全静态链接）"
    else
        echo -e "${RED}构建失败！${NC}"
        exit 1
    fi
}

# ==================== 主入口 ====================
case "${1:-dynamic}" in
    dynamic)
        build_dynamic
        ;;
    static)
        build_static
        ;;
    minimal)
        build_minimal
        ;;
    -h | --help | help)
        print_usage
        ;;
    *)
        echo -e "${RED}未知选项: $1${NC}"
        print_usage
        exit 1
        ;;
esac
