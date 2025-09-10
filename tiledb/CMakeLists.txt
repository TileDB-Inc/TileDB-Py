# Pybind11

# Enable @rpath support on macOS for proper library isolation
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

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

target_compile_features(
    main
    PUBLIC
    cxx_std_20
)

if (TILEDB_SERIALIZATION)
    target_compile_definitions(
        main
        PRIVATE
        TILEDB_SERIALIZATION
    )
endif()

install(TARGETS main DESTINATION tiledb)

# Handle TileDB shared library installation and RPATH setup
if(TILEDB_DOWNLOADED)
    if(ENV{CI_WHEEL_BUILD})
        # For CI builds - install to external directory
        set(TILEDB_EXTERNAL_DIR "$ENV{RUNNER_TEMP}/tiledb-external")
        message(STATUS "CI Build - installing TileDB to: ${TILEDB_EXTERNAL_DIR}")
        install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared 
                DESTINATION "${TILEDB_EXTERNAL_DIR}")
        
        # Set RPATH for CI builds
        if(APPLE)
            set_target_properties(main PROPERTIES 
                INSTALL_RPATH "@rpath"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(main PROPERTIES 
                INSTALL_RPATH "${TILEDB_EXTERNAL_DIR}"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    else()
        # For local builds - install to tiledb directory
        message(STATUS "Local Build - installing TileDB shared library to tiledb directory")
        install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared DESTINATION tiledb)
        
        # Set RPATH for local builds
        if(APPLE)
            set_target_properties(main PROPERTIES 
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        elseif(UNIX)
            set_target_properties(main PROPERTIES 
                INSTALL_RPATH "\$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
else()
    # For external TileDB installations - no installation needed, just set RPATH
    get_property(TILEDB_LOCATION TARGET TileDB::tiledb_shared PROPERTY LOCATION)
    get_filename_component(TILEDB_LOCATION ${TILEDB_LOCATION} DIRECTORY)
    message(STATUS "External TileDB - setting RPATH to: ${TILEDB_LOCATION}")
    
    if(APPLE)
        set_target_properties(main PROPERTIES 
            INSTALL_RPATH "@rpath"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(main PROPERTIES 
            INSTALL_RPATH ${TILEDB_LOCATION}
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

add_subdirectory(libtiledb)