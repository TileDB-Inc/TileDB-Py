#include "common.h"

#include <numeric>

std::unordered_map<tiledb_datatype_t, std::string> _tdb_to_np_name_dtype = {
    {TILEDB_INT32, "int32"},
    {TILEDB_INT64, "int64"},
    {TILEDB_FLOAT32, "float32"},
    {TILEDB_FLOAT64, "float64"},
    {TILEDB_INT8, "int8"},
    {TILEDB_UINT8, "uint8"},
    {TILEDB_INT16, "int16"},
    {TILEDB_UINT16, "uint16"},
    {TILEDB_UINT32, "uint32"},
    {TILEDB_UINT64, "uint64"},
    {TILEDB_STRING_ASCII, "S"},
    {TILEDB_STRING_UTF8, "U1"},
    {TILEDB_CHAR, "S1"},
    {TILEDB_DATETIME_YEAR, "M8[Y]"},
    {TILEDB_DATETIME_MONTH, "M8[M]"},
    {TILEDB_DATETIME_WEEK, "M8[W]"},
    {TILEDB_DATETIME_DAY, "M8[D]"},
    {TILEDB_DATETIME_HR, "M8[h]"},
    {TILEDB_DATETIME_MIN, "M8[m]"},
    {TILEDB_DATETIME_SEC, "M8[s]"},
    {TILEDB_DATETIME_MS, "M8[ms]"},
    {TILEDB_DATETIME_US, "M8[us]"},
    {TILEDB_DATETIME_NS, "M8[ns]"},
    {TILEDB_DATETIME_PS, "M8[ps]"},
    {TILEDB_DATETIME_FS, "M8[fs]"},
    {TILEDB_DATETIME_AS, "M8[as]"},
    /* duration types map to timedelta */
    {TILEDB_TIME_HR, "m8[h]"},
    {TILEDB_TIME_MIN, "m8[m]"},
    {TILEDB_TIME_SEC, "m8[s]"},
    {TILEDB_TIME_MS, "m8[ms]"},
    {TILEDB_TIME_US, "m8[us]"},
    {TILEDB_TIME_NS, "m8[ns]"},
    {TILEDB_TIME_PS, "m8[ps]"},
    {TILEDB_TIME_FS, "m8[fs]"},
    {TILEDB_TIME_AS, "m8[as]"},
    {TILEDB_BLOB, "byte"},
    {TILEDB_BOOL, "bool"},
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 21
    {TILEDB_GEOM_WKB, "byte"},
    {TILEDB_GEOM_WKT, "S"},
#endif
};

std::unordered_map<std::string, tiledb_datatype_t> _np_name_to_tdb_dtype = {
    {"int32", TILEDB_INT32},
    {"int64", TILEDB_INT64},
    {"float32", TILEDB_FLOAT32},
    {"float64", TILEDB_FLOAT64},
    {"int8", TILEDB_INT8},
    {"uint8", TILEDB_UINT8},
    {"int16", TILEDB_INT16},
    {"uint16", TILEDB_UINT16},
    {"uint32", TILEDB_UINT32},
    {"uint64", TILEDB_UINT64},
    {"datetime64[Y]", TILEDB_DATETIME_YEAR},
    {"datetime64[M]", TILEDB_DATETIME_MONTH},
    {"datetime64[W]", TILEDB_DATETIME_WEEK},
    {"datetime64[D]", TILEDB_DATETIME_DAY},
    {"datetime64[h]", TILEDB_DATETIME_HR},
    {"datetime64[m]", TILEDB_DATETIME_MIN},
    {"datetime64[s]", TILEDB_DATETIME_SEC},
    {"datetime64[ms]", TILEDB_DATETIME_MS},
    {"datetime64[us]", TILEDB_DATETIME_US},
    {"datetime64[ns]", TILEDB_DATETIME_NS},
    {"datetime64[ps]", TILEDB_DATETIME_PS},
    {"datetime64[fs]", TILEDB_DATETIME_FS},
    {"datetime64[as]", TILEDB_DATETIME_AS},
    /* duration types map to timedelta */
    {"timedelta64[h]", TILEDB_TIME_HR},
    {"timedelta64[m]", TILEDB_TIME_MIN},
    {"timedelta64[s]", TILEDB_TIME_SEC},
    {"timedelta64[ms]", TILEDB_TIME_MS},
    {"timedelta64[us]", TILEDB_TIME_US},
    {"timedelta64[ns]", TILEDB_TIME_NS},
    {"timedelta64[ps]", TILEDB_TIME_PS},
    {"timedelta64[fs]", TILEDB_TIME_FS},
    {"timedelta64[as]", TILEDB_TIME_AS},
    {"bool", TILEDB_BOOL},
};

namespace tiledbpy::common {

size_t buffer_nbytes(py::buffer_info &info) {
  return info.itemsize * std::accumulate(info.shape.begin(), info.shape.end(),
                                         1, std::multiplies<>());
}

bool expect_buffer_nbytes(py::buffer_info &info, tiledb_datatype_t datatype,
                          size_t nelem) {
  size_t nbytes = buffer_nbytes(info);
  size_t nbytes_expected = tiledb_datatype_size(datatype) * nelem;
  return nbytes == nbytes_expected;
}

} // namespace tiledbpy::common

py::dtype tdb_to_np_dtype(tiledb_datatype_t type, uint32_t cell_val_num) {
  if (type == TILEDB_CHAR || type == TILEDB_STRING_UTF8 ||
      type == TILEDB_STRING_ASCII) {
    std::string base_str = (type == TILEDB_STRING_UTF8) ? "|U" : "|S";
    if (cell_val_num < TILEDB_VAR_NUM)
      base_str += std::to_string(cell_val_num);
    return py::dtype(base_str);
  }

  if (cell_val_num == 1) {
    if (type == TILEDB_STRING_UTF16 || type == TILEDB_STRING_UTF32)
      TPY_ERROR_LOC("Unimplemented UTF16 or UTF32 string conversion!");
    if (type == TILEDB_STRING_UCS2 || type == TILEDB_STRING_UCS4)
      TPY_ERROR_LOC("Unimplemented UCS2 or UCS4 string conversion!");

    if (_tdb_to_np_name_dtype.count(type) == 1)
      return py::dtype(_tdb_to_np_name_dtype[type]);
  }

  if (cell_val_num == 2) {
    if (type == TILEDB_FLOAT32)
      return py::dtype("complex64");
    if (type == TILEDB_FLOAT64)
      return py::dtype("complex128");
  }

  if (cell_val_num == TILEDB_VAR_NUM)
    return tdb_to_np_dtype(type, 1);

  if (cell_val_num > 1) {
    py::dtype base_dtype = tdb_to_np_dtype(type, 1);
    py::tuple rec_elem = py::make_tuple("", base_dtype);
    py::list rec_list;
    for (size_t i = 0; i < cell_val_num; i++)
      rec_list.append(rec_elem);
    // note: we call the 'dtype' constructor b/c py::dtype does not accept
    // list
    auto np = py::module::import("numpy");
    auto np_dtype = np.attr("dtype");
    return np_dtype(rec_list);
  }

  TPY_ERROR_LOC("tiledb datatype not understood ('" +
                tiledb::impl::type_to_str(type) +
                "', cell_val_num: " + std::to_string(cell_val_num) + ")");
}

tiledb_datatype_t np_to_tdb_dtype(py::dtype type) {
  auto name = py::str(py::getattr(type, "name"));
  if (_np_name_to_tdb_dtype.count(name) == 1)
    return _np_name_to_tdb_dtype[name];

  auto kind = py::str(py::getattr(type, "kind"));
  if (kind == py::str("S"))
    return TILEDB_STRING_ASCII;
  if (kind == py::str("U"))
    return TILEDB_STRING_UTF8;

  TPY_ERROR_LOC("could not handle numpy dtype: " +
                py::getattr(type, "name").cast<std::string>());
}

bool is_tdb_num(tiledb_datatype_t type) {
  switch (type) {
  case TILEDB_INT8:
  case TILEDB_INT16:
  case TILEDB_UINT8:
  case TILEDB_INT32:
  case TILEDB_INT64:
  case TILEDB_UINT16:
  case TILEDB_UINT32:
  case TILEDB_UINT64:
  case TILEDB_FLOAT32:
  case TILEDB_FLOAT64:
    return true;
  default:
    return false;
  }
}

bool is_tdb_str(tiledb_datatype_t type) {
  switch (type) {
  case TILEDB_STRING_ASCII:
  case TILEDB_STRING_UTF8:
  case TILEDB_CHAR:
    return true;
  default:
    return false;
  }
}

py::size_t get_ncells(py::dtype type) {
  if (type == py::dtype("S"))
    return type.itemsize() == 0 ? TILEDB_VAR_NUM : type.itemsize();

  if (type.is(py::dtype("U"))) {
    auto np_unicode_size = py::dtype("U").itemsize();
    return type.itemsize() == 0 ? TILEDB_VAR_NUM
                                : type.itemsize() / np_unicode_size;
  }

  auto np = py::module::import("numpy");
  auto np_issubdtype = np.attr("issubdtype");
  auto np_complexfloating = np.attr("complexfloating");
  py::bool_ iscomplexfloating = np_issubdtype(type, np_complexfloating);
  if (iscomplexfloating)
    return 2;

  return 1;
}

py::array_t<uint8_t>
uint8_bool_to_uint8_bitmap(py::array_t<uint8_t> validity_array) {
  // TODO profile, probably replace; avoid inplace reassignment
  auto np = py::module::import("numpy");
  auto packbits = np.attr("packbits");
  auto tmp = packbits(validity_array, "bitorder"_a = "little");
  return tmp;
}

uint64_t count_zeros(py::array_t<uint8_t> a) {
  uint64_t count = 0;
  for (py::ssize_t idx = 0; idx < a.size(); idx++)
    count += (a.data()[idx] == 0) ? 1 : 0;
  return count;
}
