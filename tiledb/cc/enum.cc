#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void init_enums(py::module &m) {
  // consts from tiledb.h
  m.def("TILEDB_VAR_NUM", []() { return TILEDB_VAR_NUM; });
  m.def("TILEDB_MAX_PATH", []() { return TILEDB_MAX_PATH; });
  m.def("TILEDB_OFFSET_SIZE", []() { return TILEDB_OFFSET_SIZE; });
  m.def("TILEDB_TIMESTAMP_NOW_MS", []() { return TILEDB_TIMESTAMP_NOW_MS; });

  py::enum_<tiledb_datatype_t>(m, "DataType", py::module_local())
      .value("INT32", TILEDB_INT32)
      .value("INT64", TILEDB_INT64)
      .value("FLOAT32", TILEDB_FLOAT32)
      .value("FLOAT64", TILEDB_FLOAT64)
      .value("CHAR", TILEDB_CHAR)
      .value("INT8", TILEDB_INT8)
      .value("UINT8", TILEDB_UINT8)
      .value("INT16", TILEDB_INT16)
      .value("UINT16", TILEDB_UINT16)
      .value("UINT32", TILEDB_UINT32)
      .value("UINT64", TILEDB_UINT64)
      .value("BOOL", TILEDB_BOOL)
      .value("STRING_ASCII", TILEDB_STRING_ASCII)
      .value("STRING_UTF8", TILEDB_STRING_UTF8)
      .value("STRING_UTF16", TILEDB_STRING_UTF16)
      .value("STRING_UTF32", TILEDB_STRING_UTF32)
      .value("STRING_UCS2", TILEDB_STRING_UCS2)
      .value("STRING_UCS4", TILEDB_STRING_UCS4)
      .value("ANY", TILEDB_ANY)
      .value("DATETIME_YEAR", TILEDB_DATETIME_YEAR)
      .value("DATETIME_MONTH", TILEDB_DATETIME_MONTH)
      .value("DATETIME_WEEK", TILEDB_DATETIME_WEEK)
      .value("DATETIME_DAY", TILEDB_DATETIME_DAY)
      .value("DATETIME_HR", TILEDB_DATETIME_HR)
      .value("DATETIME_MIN", TILEDB_DATETIME_MIN)
      .value("DATETIME_SEC", TILEDB_DATETIME_SEC)
      .value("DATETIME_MS", TILEDB_DATETIME_MS)
      .value("DATETIME_US", TILEDB_DATETIME_US)
      .value("DATETIME_NS", TILEDB_DATETIME_NS)
      .value("DATETIME_PS", TILEDB_DATETIME_PS)
      .value("DATETIME_FS", TILEDB_DATETIME_FS)
      .value("DATETIME_AS", TILEDB_DATETIME_AS)
      .value("TIME_HR", TILEDB_TIME_HR)
      .value("TIME_MIN", TILEDB_TIME_MIN)
      .value("TIME_SEC", TILEDB_TIME_SEC)
      .value("TIME_MS", TILEDB_TIME_MS)
      .value("TIME_US", TILEDB_TIME_US)
      .value("TIME_NS", TILEDB_TIME_NS)
      .value("TIME_PS", TILEDB_TIME_PS)
      .value("TIME_FS", TILEDB_TIME_FS)
      .value("TIME_AS", TILEDB_TIME_AS)
      .value("BLOB", TILEDB_BLOB)
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 21
      .value("GEOM_WKB", TILEDB_GEOM_WKB)
      .value("GEOM_WKT", TILEDB_GEOM_WKT)
#endif
      ; // line continuation for ifdef

  py::enum_<tiledb_array_type_t>(m, "ArrayType")
      .value("DENSE", TILEDB_DENSE)
      .value("SPARSE", TILEDB_SPARSE);

  py::enum_<tiledb_layout_t>(m, "LayoutType")
      .value("ROW_MAJOR", TILEDB_ROW_MAJOR)
      .value("COL_MAJOR", TILEDB_COL_MAJOR)
      .value("GLOBAL_ORDER", TILEDB_GLOBAL_ORDER)
      .value("UNORDERED", TILEDB_UNORDERED)
      .value("HILBERT", TILEDB_HILBERT);

  py::enum_<tiledb_data_order_t>(m, "DataOrder")
      .value("UNORDERED_DATA", TILEDB_UNORDERED_DATA)
      .value("INCREASING_DATA", TILEDB_INCREASING_DATA)
      .value("DECREASING_DATA", TILEDB_DECREASING_DATA);

  py::enum_<tiledb_filter_type_t>(m, "FilterType")
      .value("NONE", TILEDB_FILTER_NONE)
      .value("GZIP", TILEDB_FILTER_GZIP)
      .value("ZSTD", TILEDB_FILTER_ZSTD)
      .value("LZ4", TILEDB_FILTER_LZ4)
      .value("RLE", TILEDB_FILTER_RLE)
      .value("BZIP2", TILEDB_FILTER_BZIP2)
      .value("DELTA", TILEDB_FILTER_DELTA)
      .value("DOUBLE_DELTA", TILEDB_FILTER_DOUBLE_DELTA)
      .value("BIT_WIDTH_REDUCTION", TILEDB_FILTER_BIT_WIDTH_REDUCTION)
      .value("BITSHUFFLE", TILEDB_FILTER_BITSHUFFLE)
      .value("BYTESHUFFLE", TILEDB_FILTER_BYTESHUFFLE)
      .value("POSITIVE_DELTA", TILEDB_FILTER_POSITIVE_DELTA)
      .value("CHECKSUM_MD5", TILEDB_FILTER_CHECKSUM_MD5)
      .value("CHECKSUM_SHA256", TILEDB_FILTER_CHECKSUM_SHA256)
      .value("SCALE_FLOAT", TILEDB_FILTER_SCALE_FLOAT)
      .value("DICTIONARY", TILEDB_FILTER_DICTIONARY)
      .value("XOR", TILEDB_FILTER_XOR)
      .value("WEBP", TILEDB_FILTER_WEBP);

  py::enum_<tiledb_filter_option_t>(m, "FilterOption")
      .value("COMPRESSION_LEVEL", TILEDB_COMPRESSION_LEVEL)
      .value("BIT_WIDTH_MAX_WINDOW", TILEDB_BIT_WIDTH_MAX_WINDOW)
      .value("POSITIVE_DELTA_MAX_WINDOW", TILEDB_POSITIVE_DELTA_MAX_WINDOW)
      .value("SCALE_FLOAT_BYTEWIDTH", TILEDB_SCALE_FLOAT_BYTEWIDTH)
      .value("SCALE_FLOAT_FACTOR", TILEDB_SCALE_FLOAT_FACTOR)
      .value("SCALE_FLOAT_OFFSET", TILEDB_SCALE_FLOAT_OFFSET)
      .value("WEBP_INPUT_FORMAT", TILEDB_WEBP_INPUT_FORMAT)
      .value("WEBP_QUALITY", TILEDB_WEBP_QUALITY)
      .value("WEBP_LOSSLESS", TILEDB_WEBP_LOSSLESS)
      .value("COMPRESSION_REINTERPRET_DATATYPE",
             TILEDB_COMPRESSION_REINTERPRET_DATATYPE);

  py::enum_<tiledb_filter_webp_format_t>(m, "WebpInputFormat")
      .value("WEBP_NONE", TILEDB_WEBP_NONE)
      .value("WEBP_RGB", TILEDB_WEBP_RGB)
      .value("WEBP_RGBA", TILEDB_WEBP_RGBA)
      .value("WEBP_BGR", TILEDB_WEBP_BGR)
      .value("WEBP_BGRA", TILEDB_WEBP_BGRA);

  py::enum_<tiledb_encryption_type_t>(m, "EncryptionType")
      .value("NO_ENCRYPTION", TILEDB_NO_ENCRYPTION)
      .value("AES_256_GCM", TILEDB_AES_256_GCM);

  py::enum_<tiledb::Query::Status>(m, "QueryStatus")
      .value("FAILED", Query::Status::FAILED)
      .value("COMPLETE", Query::Status::COMPLETE)
      .value("INPROGRESS", Query::Status::INPROGRESS)
      .value("INCOMPLETE", Query::Status::INCOMPLETE)
      .value("UNINITIALIZED", Query::Status::UNINITIALIZED)
      .export_values();

  py::enum_<tiledb_query_type_t>(m, "QueryType")
      .value("READ", TILEDB_READ)
      .value("WRITE", TILEDB_WRITE)
      .value("DELETE", TILEDB_DELETE)
      .value("MODIFY_EXCLUSIVE", TILEDB_MODIFY_EXCLUSIVE);

  py::enum_<tiledb_query_condition_op_t>(m, "QueryConditionOp",
                                         py::module_local())
      .value("LT", TILEDB_LT)
      .value("LE", TILEDB_LE)
      .value("GT", TILEDB_GT)
      .value("GE", TILEDB_GE)
      .value("EQ", TILEDB_EQ)
      .value("NE", TILEDB_NE);

  py::enum_<tiledb_vfs_mode_t>(m, "VFSMode")
      .value("READ", TILEDB_VFS_READ)
      .value("WRITE", TILEDB_VFS_WRITE)
      .value("APPEND", TILEDB_VFS_APPEND);

  py::enum_<tiledb_filesystem_t>(m, "FileSystem")
      .value("S3", TILEDB_S3)
      .value("AZURE", TILEDB_AZURE)
      .value("GCS", TILEDB_GCS)
      .value("HDFS", TILEDB_HDFS);

  py::enum_<tiledb::Object::Type>(m, "ObjectType")
      .value("ARRAY", Object::Type::Array)
      .value("GROUP", Object::Type::Group)
      .value("INVALID", Object::Type::Invalid)
      .export_values();

  py::enum_<tiledb_mime_type_t>(m, "MIMEType")
      .value("AUTODETECT", TILEDB_MIME_AUTODETECT)
      .value("TIFF", TILEDB_MIME_TIFF)
      .value("PDF", TILEDB_MIME_PDF);

  // test helpers to check enum name against typed value
  m.def("_enum_string", &tiledb::impl::type_to_str);
  m.def("_enum_string",
        py::overload_cast<tiledb_array_type_t>(&tiledb::ArraySchema::to_str));
  m.def("_enum_string",
        py::overload_cast<tiledb_layout_t>(&tiledb::ArraySchema::to_str));
  m.def("_enum_string",
        py::overload_cast<tiledb_filter_type_t>(&tiledb::Filter::to_str));
  m.def("_enum_string", [](Query::Status status) {
    std::stringstream ss;
    ss << status;
    return ss.str();
  });
}

}; // namespace libtiledbcpp
