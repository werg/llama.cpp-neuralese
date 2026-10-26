# natlang native state spans: forward-parity dump against natlang's reference engine (scripts/llama_state_span_parity.py)

set(TARGET llama-state-span-dump)
add_executable(${TARGET} state-span-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common vendor::nlohmann)
