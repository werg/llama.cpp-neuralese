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
