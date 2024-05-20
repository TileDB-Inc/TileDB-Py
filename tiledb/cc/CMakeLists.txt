pybind11_add_module(
    cc
    array.cc
    attribute.cc
    common.cc
    common.h
    consolidation_plan.cc
    context.cc
    dimension_label.cc
    domain.cc
    enum.cc
    enumeration.cc
    filestore.cc
    filter.cc
    group.cc
    object.cc
    query.cc
    schema.cc
    subarray.cc
    tiledbcpp.cc
    vfs.cc
)

target_link_libraries(
    cc
    PUBLIC
    TileDB::tiledb_shared
)

target_compile_features(
    cc
    PUBLIC
    cxx_std_20
)

install(TARGETS cc DESTINATION tiledb)

if(TILEDB_DOWNLOADED)
    if (APPLE)
    set_target_properties(cc PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
    set_target_properties(cc PROPERTIES INSTALL_RPATH "\$ORIGIN")
    endif()
endif()
