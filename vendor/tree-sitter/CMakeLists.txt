# tree-sitter 0.26.0 and its TypeScript grammar 0.23.2 (the versions natlang's reference server uses),
# for the server's guided generation (tools/server/server-guidance.cpp)
add_library(tree-sitter-typescript STATIC
    tree-sitter/src/lib.c
    tree-sitter-typescript/typescript/src/parser.c
    tree-sitter-typescript/typescript/src/scanner.c)
add_library(vendor::tree-sitter-typescript ALIAS tree-sitter-typescript)

target_include_directories(tree-sitter-typescript PUBLIC tree-sitter/include PRIVATE tree-sitter/src tree-sitter-typescript/typescript/src)
set_target_properties(tree-sitter-typescript PROPERTIES C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)
target_compile_options(tree-sitter-typescript PRIVATE -w) # vendored code
