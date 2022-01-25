#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

#define DENUM(x) .value(#x, TILEDB_##x)

void init_enums(py::module& m) {
  py::enum_<tiledb_datatype_t>(m, "DataType")
    DENUM(INT32)
    DENUM(INT64)
    DENUM(FLOAT32)
    DENUM(FLOAT64)
    DENUM(CHAR)
    DENUM(INT8)
    DENUM(UINT8)
    DENUM(INT16)
    DENUM(UINT16)
    DENUM(UINT32)
    DENUM(UINT64)
    DENUM(STRING_ASCII)
    DENUM(STRING_UTF8)
    DENUM(STRING_UTF16)
    DENUM(STRING_UTF32)
    DENUM(STRING_UCS2)
    DENUM(STRING_UCS4)
    DENUM(ANY)
    DENUM(DATETIME_YEAR)
    DENUM(DATETIME_WEEK)
    DENUM(DATETIME_DAY)
    DENUM(DATETIME_HR)
    DENUM(DATETIME_MIN)
    DENUM(DATETIME_SEC)
    DENUM(DATETIME_MS)
    DENUM(DATETIME_US)
    DENUM(DATETIME_NS)
    DENUM(DATETIME_PS)
    DENUM(DATETIME_FS)
    DENUM(DATETIME_AS)
    DENUM(TIME_HR)
    DENUM(TIME_MIN)
    DENUM(TIME_SEC)
    DENUM(TIME_MS)
    DENUM(TIME_US)
    DENUM(TIME_NS)
    DENUM(TIME_PS)
    DENUM(TIME_FS)
    DENUM(TIME_AS);

  py::enum_<tiledb_array_type_t>(m, "ArrayType")
    DENUM(DENSE)
    DENUM(SPARSE);

  py::enum_<tiledb_layout_t>(m, "LayoutType")
    DENUM(ROW_MAJOR)
    DENUM(COL_MAJOR)
    DENUM(UNORDERED)
    DENUM(HILBERT);

  py::enum_<tiledb_filter_type_t>(m, "FilterType")
    DENUM(FILTER_NONE)
    DENUM(FILTER_GZIP)
    DENUM(FILTER_ZSTD)
    DENUM(FILTER_LZ4)
    DENUM(FILTER_RLE)
    DENUM(FILTER_BZIP2)
    DENUM(FILTER_DOUBLE_DELTA)
    DENUM(FILTER_BIT_WIDTH_REDUCTION)
    DENUM(FILTER_BITSHUFFLE)
    DENUM(FILTER_BYTESHUFFLE)
    DENUM(FILTER_POSITIVE_DELTA)
    DENUM(FILTER_CHECKSUM_MD5)
    DENUM(FILTER_CHECKSUM_SHA256);

  py::enum_<tiledb_filter_option_t>(m, "FilterOption")
    DENUM(COMPRESSION_LEVEL)
    DENUM(BIT_WIDTH_MAX_WINDOW)
    DENUM(POSITIVE_DELTA_MAX_WINDOW);

  py::enum_<tiledb_encryption_type_t>(m, "EncryptionType")
    DENUM(NO_ENCRYPTION)
    DENUM(AES_256_GCM);

  py::enum_<tiledb_query_status_t>(m, "QueryStatus")
    DENUM(FAILED)
    DENUM(COMPLETED)
    DENUM(INPROGRESS)
    DENUM(INCOMPLETE)
    DENUM(UNINITIALIZED);

  py::enum_<tiledb_query_condition_op_t>(m, "QueryConditionOp")
    DENUM(LT)
    DENUM(LE)
    DENUM(GT)
    DENUM(GE)
    DENUM(EQ)
    DENUM(NE);


  // test helpers to check enum name against typed value
  m.def("_enum_string", &tiledb::impl::type_to_str);
  m.def("_enum_string",
        py::overload_cast<tiledb_array_type_t>(&tiledb::ArraySchema::to_str));
  m.def("_enum_string",
        py::overload_cast<tiledb_layout_t>(&tiledb::ArraySchema::to_str));
}

}; // namespace libtiledbcpp