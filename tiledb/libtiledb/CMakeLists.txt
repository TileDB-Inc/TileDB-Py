# Enable @rpath support on macOS for proper library isolation
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

pybind11_add_module(
    libtiledb
    array.cc
    attribute.cc
    common.cc
    common.h
    consolidation_plan.cc
    context.cc
    current_domain.cc
    dimension_label.cc
    domain.cc
    enum.cc
    enumeration.cc
    filestore.cc
    filter.cc
    group.cc
    metadata.h
    object.cc
    profile.cc
    query.cc
    schema.cc
    subarray.cc
    tiledbcpp.cc
    vfs.cc
)

target_link_libraries(
    libtiledb
    PUBLIC
    TileDB::tiledb_shared
)

target_compile_features(
    libtiledb
    PUBLIC
    cxx_std_20
)

if (TILEDB_REMOVE_DEPRECATIONS)
    target_compile_definitions(
        libtiledb
        PRIVATE
        TILEDB_REMOVE_DEPRECATIONS
    )
endif()

if (TILEDB_SERIALIZATION)
    target_compile_definitions(
        libtiledb
        PRIVATE
        TILEDB_SERIALIZATION
    )
endif()

install(TARGETS libtiledb DESTINATION tiledb)

if(NOT TILEDB_DOWNLOADED)    
    # If using external TileDB core library force it to be linked at runtime using RPATH
    get_property(TILEDB_LOCATION TARGET TileDB::tiledb_shared PROPERTY LOCATION)
    get_filename_component(TILEDB_LOCATION ${TILEDB_LOCATION} DIRECTORY)
    message(STATUS "Setting RPATH for target \"libtiledb\" to ${TILEDB_LOCATION}")
    
    if(APPLE)
        set_target_properties(libtiledb PROPERTIES 
            INSTALL_RPATH "@rpath"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(libtiledb PROPERTIES INSTALL_RPATH ${TILEDB_LOCATION})
    endif()
endif()
