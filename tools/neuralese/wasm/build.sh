#!/bin/sh
# Build the Neuralese service as WebAssembly (tools/neuralese/wasm/neuralese-wasm.{js,wasm} in build-wasm/bin) with
# the Emscripten SDK image. Single-threaded CPU (SIMD); WebGPU and threads come later.
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
IMAGE=${EMSDK_IMAGE:-emscripten/emsdk:4.0.20-arm64}
docker run --rm -u "$(id -u):$(id -g)" -v "$ROOT":/src -w /src "$IMAGE" sh -c '
  emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
    -DCMAKE_C_FLAGS="-msimd128 -fwasm-exceptions" -DCMAKE_CXX_FLAGS="-msimd128 -fwasm-exceptions" \
    -DCMAKE_EXE_LINKER_FLAGS="-fwasm-exceptions" >/dev/null &&
  cmake --build build-wasm --target neuralese-wasm -j "$(nproc)"'
ls -la "$ROOT"/build-wasm/bin/neuralese-wasm*
