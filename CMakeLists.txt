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

if (TILEDB_BRANCH)
  include(FetchContent)
  message(STATUS "Building TileDB from source branch \"${TILEDB_BRANCH}\"")

  FetchContent_Declare(
    TileDB
    GIT_REPOSITORY https://github.com/TileDB-Inc/TileDB.git
    GIT_TAG ${TILEDB_BRANCH}
  )

  FetchContent_MakeAvailable(TileDB)
  set(TILEDB_DOWNLOADED TRUE)
elseif (NOT TileDB_FOUND)
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
                VERSION 2.27.2
                RELLIST_HASH SHA256=b514c20cfd45c2250fea5c2efcb80facf953e647416fc1fdc6ac8133a7f8dc4d
        )
    endif()
    find_package(TileDB REQUIRED)
    set(TILEDB_DOWNLOADED TRUE)
else()
    message(STATUS "Found external TileDB core library")
endif()

message(STATUS "Using libtiledb path: " "${TileDB_DIR}")
add_subdirectory(tiledb)
