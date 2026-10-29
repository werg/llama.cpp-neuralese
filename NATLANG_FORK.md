# natlang fork of llama.cpp

This is natlang's fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). It serves natlang's native models,
the LFM2/LFM3 lines with native Neuralese state, and keeps upstream behaviour unchanged everywhere else. The fork is
merged with upstream regularly rather than rebased, so the history of our changes is kept.

- Branch: `neuralese` (default). The upstream base of the latest merge is in `git log --merges --first-parent`; the
  merge of 2026-10-29 brought in upstream master `0893ec98f5eafad36a43a72cd1c3723d72b0d03a`.
- The natlang repository pins the fork commit it uses in `training/neuralese/llama-cpp-fork.json`.

## What the fork adds

Detailed API documentation for the server additions is in `tools/server/README.md` (sections "Native state spans" and
"natlang API Endpoints").

- **Native state spans** (`src/llama-state-span.h`, `common/native-state.{h,cpp}`, `llama_state_span_register`): exact
  imported per-layer state (positional KV plus LFM short-convolution state) for runs of virtual positions. A GGUF span
  is loaded with `llama-server --native-state FILE`, and a chat content part `{"type": "native_state", "id": ID}`
  consumes it between its consume markers.
- **Native state production**: the producer adapter and per-sequence capture of exported state, and the native-state
  protocol in llama-server (`/v1/neuralese/info`, `/v1/neuralese/native-states`, consumption receipts).
- **Per-sequence LoRA** (`--lora-per-sequence`): each request's adapter scales affect only its own tokens, so requests
  with different adapters batch together. A prompt state computed with other adapters is never reused.
- **Decision readout** (`POST /v1/natlang/score`): scores continuations of chat prompts.
- **Guided generation** (`guidance` request field, `POST /v1/neuralese/guidance/check`): checks natlang `eval` tool
  calls while they are generated (repetition, redeclaration, TypeScript syntax via vendored tree-sitter in `vendor/`)
  and rolls back and resamples on a rejection. It supports pythonic (LFM2) and XML (LFM3) tool-call formats. Reasoning
  the generation prompt opened (`<think>` for LFM3) is not an action: guidance neither checks nor rolls it back, so a
  streamed guided reply sends its reasoning as it is generated and holds back only the rest until it is final.
- **RL actor capture** (`x_natlang_rl_capture`): exact sampled-token log-probabilities, behaviour top-k, prompt and
  native-state segments, producer writes and an actor-export attestation (including mixed quantized actors, tensor
  overrides and the running executable and library hashes) for natlang's whole-graph RL learner.
- **Backend RL capture and guidance bans** (`llama_sampler_init_dist_capture`, `llama_sampler_dist_set_bans`): on the
  backend the draw also returns the drawn token's probability and the sampler's top-k, so RL rollouts and guided
  generation keep sampling on the device instead of copying full logits per token (`tests/test-backend-sampler.cpp`
  `test_dist_capture*`).
- **Parity tools** (`tools/neuralese`): `llama-logits-dump`, which dumps all-position logits and per-layer residual
  states, and `llama-state-span-dump`, which runs a forward pass with imported state spans.
- **CPU flash attention**: softmax(KQ)·V is accumulated in FP32 for F16 V.

LFM3 itself (`Lfm3ForCausalLM`) is supported upstream; the fork adds the native-state and serving pieces above to it.

## Building

The build is upstream's CMake build. The natlang-specific targets are `llama-server`, `llama-state-span-dump` and
`llama-logits-dump`.

CPU only:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target llama-server llama-state-span-dump llama-logits-dump -j
```

CUDA: set `CMAKE_CUDA_ARCHITECTURES` for the target GPU. Examples are `121` for DGX Spark / GB10, `120a-real` for RTX
PRO 6000 Blackwell, `100a-real` for B200 and `90a-real` for H100.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real
cmake --build build-cuda --target llama-server llama-state-span-dump llama-logits-dump -j
```

- For a binary to copy to other machines, add `-DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF`. This links ggml/llama
  statically and avoids host-specific CPU flags.
- Add `-DLLAMA_BUILD_TESTS=ON` and run `ctest --test-dir build-cuda` for upstream's tests.

Example LFM3 server:

```sh
build-cuda/bin/llama-server -m LFM3-1.7B-BF16.gguf --jinja -ngl 999 -c 32768 --port 8080
```

## natlang's runtime releases

`.github/workflows/natlang-runtime.yml` builds `llama-server` and the neuralese tools for Linux x64/arm64, macOS
arm64/x64 and Windows x64/arm64 with upstream's release flags and publishes one release with `natlang-runtime.json`
(every archive's size and SHA-256). Start it with a `natlang-b<build>` tag or by hand; natlang then pins it with
`node scripts/update_llama_runtime.mjs --fork natlang-b<build>` in the natlang repository.

## Updating from upstream

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # once
git fetch upstream master
git merge upstream/master        # resolve conflicts keeping both sides; then build and test
```
