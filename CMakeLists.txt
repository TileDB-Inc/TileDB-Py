cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME})

set(PYBIND11_NEWPYTHON ON)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED
)

find_package(pybind11 CONFIG REQUIRED)

find_package(TileDB REQUIRED)

add_subdirectory(tiledb)