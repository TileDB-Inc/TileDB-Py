# Pybind11
pybind11_add_module(
    main
    main.cc
    core.cc
    npbuffer.cc
    fragment.cc
    schema_evolution.cc
    util.cc
    tests/test_metadata.cc
    tests/test_webp.cc
)

if (TILEDB_SERIALIZATION)
    target_sources(main PRIVATE serialization.cc tests/test_serialization.cc)
endif()

target_link_libraries(
    main
    PUBLIC
    TileDB::tiledb_shared
)

target_compile_features(main PUBLIC cxx_std_20)

if (TILEDB_SERIALIZATION)
    target_compile_definitions(main PRIVATE TILEDB_SERIALIZATION)
endif()

install(TARGETS main DESTINATION tiledb)
