#!/bin/sh
# Build the Neuralese service as WebAssembly (neuralese-wasm.{mjs,wasm} in build-wasm/bin; wasm32 for Safari) with
# the Emscripten SDK image. CPU (SIMD); THREADS=1 builds neuralese-wasm-mt (pthreads) in build-wasm-mt instead;
# GPU=1 builds neuralese-wasm-gpu (WebGPU through emdawnwebgpu, JSPI; Chromium) in build-wasm-gpu.
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NZ_BUILD_DIR=build-wasm; FLAGS="-msimd128 -fwasm-exceptions"; EXTRA=""
if [ "${THREADS:-0}" = 1 ]; then NZ_BUILD_DIR=build-wasm-mt; FLAGS="$FLAGS -pthread"; EXTRA="-DNZ_WASM_THREADS=ON"; fi
if [ "${GPU:-0}" = 1 ]; then NZ_BUILD_DIR=build-wasm-gpu; EXTRA="$EXTRA -DGGML_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON"; fi
IMAGE=${EMSDK_IMAGE:-emscripten/emsdk:4.0.20-arm64}
docker run --rm -u "$(id -u):$(id -g)" -v "$ROOT":/src -w /src -e NZ_BUILD_DIR="$NZ_BUILD_DIR" -e FLAGS="$FLAGS" -e EXTRA="$EXTRA" "$IMAGE" sh -c '
  emcmake cmake -S . -B $NZ_BUILD_DIR $EXTRA -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_WASM_MEM64=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
    -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" -DCMAKE_EXE_LINKER_FLAGS="$FLAGS" >/dev/null &&
  cmake --build $NZ_BUILD_DIR --target neuralese-wasm -j "$(nproc)"'
ls -la "$ROOT"/$NZ_BUILD_DIR/bin/neuralese-wasm*
