# natlang native state spans: forward-parity dumps against natlang's reference engine
# (scripts/neuralese_data/check_llama_*_parity.py in natlang)

set(TARGET llama-state-span-dump)
add_executable(${TARGET} state-span-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common vendor::nlohmann)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-logits-dump)
add_executable(${TARGET} logits-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
