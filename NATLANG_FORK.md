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

- **Native state spans** (`src/llama-state-span.h`, `llama_state_span_register`): exact imported per-layer state
  (positional KV plus LFM short-convolution state) for runs of virtual positions. A chat content part
  `{"type": "native_state", "id": ID}` consumes a stored state between its consume markers. A text write or produce
  span (`producer_kind` "text"/"span", no adapter binding or derivation) of this server's identity is consumed as its
  own tokens inline instead, with no markers (`in-context-text`; opt-outs `NATLANG_NEURALESE_TEXT_IN_CONTEXT=0`,
  `NATLANG_NEURALESE_SPAN_IN_CONTEXT=0`). Every span the server produces carries `producer_kind: "span"`.
- **Native state files** (`tools/server/server-native-state.{h,cpp}`): one format for every engine, natlang's block
  store file (`serve/store.py`: safetensors with the `natlang.native_state` metadata and the content ID). A state enters
  the server's store as received and is answered by GET as stored: a PUT body, a PUT of `{"path": FILE}` naming a store
  file on this host, a `--native-state FILE` at startup (a tuned release's `states/<id>.nzs`; the in-page engine's
  `native_states` blobs take the same path) or a produce span. Each is verified once on entry (layout, finite values,
  content ID) and decoded once for its span registration, the only computation on it. Every put after startup is one
  main-loop task (`SERVER_TASK_TYPE_NATIVE_STATE_REGISTER` carrying the bytes: verify, store, register), used by the
  HTTP PUT and by embedders; `server_context::get_native_state` reads a stored state for embedders as GET does (the
  in-page engine's `putNativeState` / `getNativeState`). The RL actor binding (the GGUF's type and hash, the process executable and
  libraries) is computed on first use (an RL capture or `/v1/neuralese/info`), not at load: a natlang GGUF loads in the
  page, which has no executable to hash, and a load no longer reads the whole model file to hash it. A model that declares its native
  identity (`natlang.identity.*`) loads only states of that identity at startup; `--native-state` needs a model with
  `natlang.consume_open/close`. The GGUF state-span format (`common/native-state.*`) is gone;
  `llama-state-span-dump` reads store files with the server's codec (`--consume OPEN CLOSE` for a stock conversion).
- **Native state production**: the producer adapter and per-sequence capture of exported state, and the native-state
  protocol in llama-server (`/v1/neuralese/info`, `/v1/neuralese/native-states`, consumption receipts). A streamed reply's
  text carries a produced state's placeholder; its last chunk carries `native_state` (blocks, imports) and
  `x_natlang_message`, the whole message with each state as a `native_state` part, as the final non-streamed reply does
  (before, a stream lost which state a placeholder named).
- **Per-sequence LoRA** (`--lora-per-sequence`): each request's adapter scales affect only its own tokens, so requests
  with different adapters batch together. A prompt state computed with other adapters is never reused.
- **Decision readout** (`POST /v1/natlang/score`): scores continuations of chat prompts, closed (the whole reply) or with
  `open_reply` as the start of the reply (natlang's field-by-field record decisions), echoed in the result; either is
  cut from the closed rendering (after the content, or after its end-of-turn token), as natlang's reference engine scores.
  With `split_tokenization` each option's probability sums its canonical tokenization and its split one (the options'
  shared leading and trailing characters tokenized apart). `reasoning_ids` scores a reasoned readout after the reasoning's sampled ids
  rather than its text re-encoded.
  An item's `reasoning` (a string) renders each continuation as the content after that reasoning (`reasoning_content`;
  LFM3 `<think>R</think>option`), so the reasoning is prompt; results echo `reasoning: true` and the decision capture
  records it.
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

`.github/workflows/natlang-runtime.yml` builds `llama-server` and the neuralese tools with upstream's release flags and
publishes one release with `natlang-runtime.json`. Backends: CPU (Linux x64/arm64, Windows x64/arm64, macOS x64), Metal
and CPU together (macOS arm64), Vulkan (Linux x64/arm64, Windows x64), ROCm 10.0 (Linux x64) and CUDA 12.4 and 13.4
(Windows x64, with a `cudart` archive each). The Windows backend archives carry the Windows CPU archive's binaries, so
each is self-contained; NVIDIA GPUs on Linux use the Vulkan archive. The same run compiles werg/wllama (the commit in
`.github/natlang-wllama.json`) with this tree as its llama.cpp to WebAssembly, so natlang's in-page engine is always the
same server code as its managed `llama-server`; a fork change that breaks the WebAssembly build fails the release.

The manifest (`natlang.llama-runtime-release/2`) lists, per archive, `file`, `os` (`linux`, `darwin`, `win32`), `arch`,
`backends` (`cpu`, `metal`, `vulkan`, `rocm`, `cuda-12.4`, `cuda-13.4`), `role` (`server`, `cudart` for the CUDA
runtime DLLs to extract next to it, or `wllama` for the in-page engine: `os` `browser`, `arch` `wasm`, with the wllama
commit it was built from), `bytes` and `sha256`. Consumers select by these fields, never by file name, and
check size and digest before extracting. Start the workflow with a `natlang-b<build>` tag or by hand. natlang pins the
release with `node scripts/update_llama_runtime.mjs --fork natlang-b<build>`; Vibestudio's local-models engine pins the
tag and the manifest's SHA-256.

## Updating from upstream

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # once
git fetch upstream master
git merge upstream/master        # resolve conflicts keeping both sides; then build and test
```
