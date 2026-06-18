#!/bin/bash
# =============================================================================
# GTCRN - 使用 Docker 构建精简版 ONNX Runtime
# =============================================================================
# 功能：
#   1. 使用 Docker 编译裁剪后的 ONNX Runtime（仅保留 GTCRN 需要的算子）
#   2. 从容器中提取编译好的静态库和头文件
#   3. 同时提取转换后的 ORT 模型格式文件
#
# 为什么要用 Docker？
#   - ONNX Runtime 的完整编译非常复杂（需要 CMake, protobuf, absl 等）
#   - Docker 提供了一致的编译环境（Ubuntu + 预装依赖）
#   - 精简后的 ORT 只有约 6MB，而完整版约 50MB+
#
# 前置条件: 已安装 Docker
# 用法: ./build-minimal-docker.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # 回到项目根目录

# 终端颜色
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }

log "正在构建 Docker 镜像（Ubuntu Rolling + ORT 最新版）..."

# 构建 Docker 镜像
# -f ladspa/Dockerfile.minimal-ort: 指定 Dockerfile
# -t gtcrn-ort-builder: 镜像名称标签
# --target export: 使用 Dockerfile 中名为 'export' 的阶段
docker build \
    -f ladspa/Dockerfile.minimal-ort \
    -t gtcrn-ort-builder \
    --target export \
    "$@" \
    .

log "从容器中提取编译好的库文件..."

# 创建临时容器（不启动，仅用于文件复制）
CONTAINER_ID=$(docker create gtcrn-ort-builder)

# 清理旧文件，确保获得最新版本
rm -rf ladspa/onnxruntime-minimal
mkdir -p ladspa/onnxruntime-minimal

# ========== 从容器复制文件 ==========
# 静态库文件（.a 文件）
docker cp "${CONTAINER_ID}:/onnxruntime-minimal/lib" ladspa/onnxruntime-minimal/
# 头文件（C++ 头文件）
docker cp "${CONTAINER_ID}:/onnxruntime-minimal/include" ladspa/onnxruntime-minimal/

# 算子配置文件和转换后的 ORT 模型
docker cp "${CONTAINER_ID}:/required_ops.config" ladspa/
# 精简模型（gtcrn_simple.onnx → gtcrn_simple.ort）
docker cp "${CONTAINER_ID}:/model.ort" stream/onnx_models/gtcrn_simple.ort
# 完整模型（gtcrn.onnx → gtcrn.ort）
docker cp "${CONTAINER_ID}:/model_full.ort" stream/onnx_models/gtcrn.ort

# 清理临时容器
docker rm "${CONTAINER_ID}" >/dev/null

success "库文件已提取到 ladspa/onnxruntime-minimal"
ls -lh ladspa/onnxruntime-minimal/lib/

echo ""
echo "下一步运行: cd ladspa && ./build.sh minimal"
