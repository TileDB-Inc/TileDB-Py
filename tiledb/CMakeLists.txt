# Cython

add_custom_command(
  OUTPUT libtiledb.cc
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/libtiledb.pyx" --output-file libtiledb.cc
  DEPENDS libtiledb.pyx
  VERBATIM
)

python_add_library(
    libtiledb
    MODULE
    libtiledb.cc
    WITH_SOABI
)

target_link_libraries(
    libtiledb
    PUBLIC
    Python::NumPy
    TileDB::tiledb_shared
)

target_compile_features(
    libtiledb
    PUBLIC
    cxx_std_20
)

# Pybind11

pybind11_add_module(
    main
    main.cc
    core.cc
    npbuffer.cc
    fragment.cc
    schema_evolution.cc
    util.cc
    serialization.cc
    tests/test_metadata.cc
    tests/test_webp.cc
    tests/test_serialization.cc
)

target_link_libraries(
    main
    PUBLIC
    TileDB::tiledb_shared
)

target_compile_features(
    main
    PUBLIC
    cxx_std_20
)

install(TARGETS main libtiledb DESTINATION tiledb)

if(TILEDB_DOWNLOADED)
    message(STATUS "Adding libtiledb.so into install group")

    install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared DESTINATION tiledb)

    if (APPLE)
        set_target_properties(main PROPERTIES INSTALL_RPATH "@loader_path")
        set_target_properties(libtiledb PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(main PROPERTIES INSTALL_RPATH "\$ORIGIN")
        set_target_properties(libtiledb PROPERTIES INSTALL_RPATH "\$ORIGIN")
    endif()
endif()

add_subdirectory(cc)