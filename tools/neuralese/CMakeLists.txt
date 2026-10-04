# neuralese: soft-token read and write ports (see neuralese.h)

add_library(neuralese
            neuralese.cpp
            neuralese.h)

target_link_libraries(neuralese PUBLIC llama ggml PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(neuralese PUBLIC .)
target_include_directories(neuralese PRIVATE ../../src)
target_compile_features(neuralese PRIVATE cxx_std_17)

set(TARGET llama-neuralese-parity)
add_executable(${TARGET} neuralese-parity.cpp)
target_link_libraries(${TARGET} PRIVATE neuralese)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# tree-sitter with its TypeScript grammar, for guided generation (vendor/: tree-sitter 0.26.0, tree-sitter-typescript
# 0.23.2, the versions natlang's reference server uses)
add_library(nz-tree-sitter STATIC vendor/tree-sitter/src/lib.c vendor/tree-sitter-typescript/typescript/src/parser.c
            vendor/tree-sitter-typescript/typescript/src/scanner.c)
target_include_directories(nz-tree-sitter PUBLIC vendor/tree-sitter/include PRIVATE vendor/tree-sitter/src
                           vendor/tree-sitter-typescript/typescript/src)
set_target_properties(nz-tree-sitter PROPERTIES C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)
target_compile_options(nz-tree-sitter PRIVATE -w) # vendored code

# The service as a WebAssembly module for natlang's browser runtime (Emscripten builds only; see wasm/build.sh)
if (EMSCRIPTEN AND TARGET llama-common)
    set(TARGET neuralese-wasm)
    add_executable(${TARGET} wasm/neuralese-wasm.cpp neuralese-service.cpp guidance.cpp)
    set_target_properties(${TARGET} PROPERTIES SUFFIX ".mjs") # an ES module (llama.cpp defaults Emscripten to .html)
    option(NZ_WASM_THREADS "the WebAssembly service with threads (needs cross-origin isolation in browsers)" OFF)
    if (NZ_WASM_THREADS)
        set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME neuralese-wasm-mt)
        target_link_options(${TARGET} PRIVATE -pthread "-sPTHREAD_POOL_SIZE=Module[\"pthreadPoolSize\"]||4")
    endif()
    target_link_libraries(${TARGET} PRIVATE neuralese nz-tree-sitter llama-common vendor::hash vendor::nlohmann)
    target_include_directories(${TARGET} PRIVATE ../../src)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    target_link_options(${TARGET} PRIVATE
        -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createNeuraleseModule -sENVIRONMENT=web,worker,node
        -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=5MB -sFORCE_FILESYSTEM=1 -lnodefs.js -lworkerfs.js
        "-sEXPORTED_FUNCTIONS=['_nzw_load','_nzw_error','_nzw_hello','_nzw_handle','_nzw_body','_nzw_body_length','_nzw_content_type','_nzw_unload','_malloc','_free']"
        "-sEXPORTED_RUNTIME_METHODS=['FS','NODEFS','WORKERFS','HEAPU8','UTF8ToString','stringToUTF8','lengthBytesUTF8']")
endif()

# OpenAI-style chat server with Neuralese blocks (same protocol as the PyTorch reference server)
if (TARGET llama-common AND TARGET cpp-httplib)
    set(TARGET llama-neuralese-server)
    add_executable(${TARGET} neuralese-server.cpp neuralese-service.cpp guidance.cpp)
    target_link_libraries(${TARGET} PRIVATE neuralese nz-tree-sitter llama-common cpp-httplib vendor::hash vendor::nlohmann
                          ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${TARGET} PRIVATE ../../src) # llama-ext.h: layer ranges, the token embedding matrix
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
