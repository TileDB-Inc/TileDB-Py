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
                VERSION 2.25.0
                RELLIST_HASH SHA256=398a1da194a59817b6f354b3232f6575c97c350d6dc792de54c529162e30763c
        )
    endif()
    find_package(TileDB REQUIRED)
    set(TILEDB_DOWNLOADED TRUE)
else()
    message(STATUS "Found external TileDB core library")
endif()

add_subdirectory(tiledb)
