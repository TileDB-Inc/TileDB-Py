cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME})

set(PYBIND11_NEWPYTHON ON)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED
)

find_package(pybind11 CONFIG REQUIRED)

if (TILEDB_PATH)
    file(REAL_PATH "${TILEDB_PATH}" USER_TILEDB_PATH EXPAND_TILDE)
    file(GLOB_RECURSE USER_TILEDB_CONFIG_PATH "${TILEDB_PATH}/**/TileDBConfig.cmake")
    cmake_path(GET USER_TILEDB_CONFIG_PATH PARENT_PATH USER_TILEDB_DIR)
    set(TileDB_DIR ${USER_TILEDB_DIR})
endif()

# In the future we should use vcpkg
if (TILEDB_PATH)
  find_package(TileDB
  HINTS ${USER_TILEDB_DIR} ${TILEDB_PATH}
  QUIET NO_DEFAULT_PATH REQUIRED)
else()
  find_package(TileDB QUIET)
endif()

if (NOT TileDB_FOUND)
    include(cmake/DownloadTileDB.cmake)
    if (TILEDB_VERSION)
        message(STATUS "Downloading TileDB version \"${TILEDB_VERSION}\" ...")
        if (TILEDB_HASH)
            fetch_prebuilt_tiledb(
                    VERSION ${TILEDB_VERSION}
                    RELLIST_HASH SHA256=${TILEDB_HASH}
            )
        else()
            fetch_prebuilt_tiledb(
                    VERSION ${TILEDB_VERSION}
            )
        endif()
    else()
        message(STATUS "Downloading TileDB default version ...")
        # Download latest release
        fetch_prebuilt_tiledb(
                VERSION 2.28.1
                RELLIST_HASH SHA256=e16928305ab3a9d48422d33de6af1b0516cf45a5cabb126465e72c0a76cbe5aa
        )
    endif()
    find_package(TileDB REQUIRED)
    set(TILEDB_DOWNLOADED TRUE)
else()
    message(STATUS "Found external TileDB core library")
endif()

message(STATUS "Using libtiledb path: " "${TileDB_DIR}")
add_subdirectory(tiledb)
