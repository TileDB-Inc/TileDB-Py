cmake_minimum_required(VERSION 3.15...3.26)

# project(${SKBUILD_PROJECT_NAME})
project(tiledb-py)

set(PYBIND11_NEWPYTHON ON)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED
)

find_package(pybind11 CONFIG REQUIRED)

# In the future we should use vcpkg
find_package(TileDB)
if (NOT TileDB_FOUND)
    include(cmake/DownloadTileDB.cmake)
    fetch_prebuilt_tiledb(
            VERSION 2.23.0
            RELLIST_HASH SHA256=2221b0ee2af2675d69f9cb31bc22062d3f379eeecc83b5fc713d00fa03d6e56f
    )
    find_package(TileDB REQUIRED)
endif()

add_subdirectory(tiledb)