cmake_minimum_required(VERSION 3.20)
project(tiledbcc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(TileDB REQUIRED PATHS /bld/TileDB-dbg/dist/lib/cmake/TileDB)
find_package(pybind11 REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

pybind11_add_module(cc
  tiledbcpp.cc
  array.cc
  context.cc
  tiledbcpp_enum.cc
  tiledbcpp_domain.cc
  query.cc
  schema.cc
)
target_link_libraries(cc PRIVATE TileDB::tiledb_shared)
