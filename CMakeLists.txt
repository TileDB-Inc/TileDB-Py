cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME})

set(PYBIND11_NEWPYTHON ON)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRE
)

find_package(pybind11 CONFIG REQUIRED)

if (TILEDB_PATH)
    # If TILEDB_PATH variable is present prepend it to the CMAKE_PREFIX_PATH
    # so that find_package will be able to find TileDB core library there
    set(CMAKE_PREFIX_PATH "${TILEDB_PATH};${CMAKE_PREFIX_PATH}")
endif()

# In the future we should use vcpkg
find_package(TileDB QUIET)
if (NOT TileDB_FOUND)
    include(cmake/DownloadTileDB.cmake)
    if (TILEDB_VERSION)
        message(STATUS "TileDB library not found, downloading version \"${TILEDB_VERSION}\" ...")
        fetch_prebuilt_tiledb(
                VERSION ${TILEDB_VERSION}
        )
    else()
        message(STATUS "TileDB library not found, downloading version default version ...")
        # Download latest release
        fetch_prebuilt_tiledb(
                VERSION 2.24.0
                RELLIST_HASH SHA256=e5aa7e6d747bb956730e438d0b29c90096226d4a553c26c5cb9161680f1d489c
        )
    endif()
    find_package(TileDB REQUIRED)
    set(TILEDB_DOWNLOADED TRUE)
endif()

add_subdirectory(tiledb)
