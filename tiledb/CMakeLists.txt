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

if(TILEDB_DOWNLOADED)

    if(DEFINED ENV{RUNNER_TEMP})
        # For CI builds
        set(TILEDB_EXTERNAL_DIR "$ENV{RUNNER_TEMP}/tiledb-external")
        message(STATUS "Installing TileDB to: ${TILEDB_EXTERNAL_DIR}")
        install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared 
                DESTINATION "${TILEDB_EXTERNAL_DIR}")
        
        # Set RPATH to use @rpath for proper library isolation on macOS
        if(APPLE)
            set_target_properties(main PROPERTIES 
                INSTALL_RPATH "@rpath"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(main PROPERTIES INSTALL_RPATH "${TILEDB_EXTERNAL_DIR}")
        endif()
    else()
        # For local builds - auditwheel is not present
        install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared DESTINATION tiledb)
        
        if (APPLE)
            set_target_properties(main PROPERTIES INSTALL_RPATH "@loader_path")
        elseif(UNIX)
            set_target_properties(main PROPERTIES INSTALL_RPATH "\$ORIGIN")
        endif()
    endif()


else()
    # If using external TileDB core library force it to be linked at runtime using RPATH
    get_property(TILEDB_LOCATION TARGET TileDB::tiledb_shared PROPERTY LOCATION)
    get_filename_component(TILEDB_LOCATION ${TILEDB_LOCATION} DIRECTORY)
    message(STATUS "Setting RPATH for targets \"main\" and \"libtiledb\" to ${TILEDB_LOCATION}")
    
    if(APPLE)
        set_target_properties(main PROPERTIES 
            INSTALL_RPATH "@rpath"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(main PROPERTIES INSTALL_RPATH ${TILEDB_LOCATION})
    endif()
endif()

add_subdirectory(libtiledb)