cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME})

set(PYBIND11_NEWPYTHON ON)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRE
)

find_package(pybind11 CONFIG REQUIRED)

# In the future we should use vcpkg
set(TileDB_DIR $ENV{TILEDB_PATH})
find_package(TileDB QUIET)
if (NOT TileDB_FOUND)
    message(STATUS "TileDB library not found, downloading...")
    include(cmake/DownloadTileDB.cmake)
    fetch_prebuilt_tiledb(
            VERSION 2.23.0
            RELLIST_HASH SHA256=2221b0ee2af2675d69f9cb31bc22062d3f379eeecc83b5fc713d00fa03d6e56f
    )
    find_package(TileDB REQUIRED)
    set(TILEDB_DOWNLOADED ON)
endif()

add_subdirectory(tiledb)