#include "common.h"

#include <numeric>

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

py::dtype tiledb_dtype(tiledb_datatype_t type, uint32_t cell_val_num) {
  if (cell_val_num == 1) {
    auto np = py::module::import("numpy");
    auto datetime64 = np.attr("datetime64");
    switch (type) {
    case TILEDB_INT32:
      return py::dtype("int32");
    case TILEDB_INT64:
      return py::dtype("int64");
    case TILEDB_FLOAT32:
      return py::dtype("float32");
    case TILEDB_FLOAT64:
      return py::dtype("float64");
    case TILEDB_INT8:
      return py::dtype("int8");
    case TILEDB_UINT8:
      return py::dtype("uint8");
    case TILEDB_INT16:
      return py::dtype("int16");
    case TILEDB_UINT16:
      return py::dtype("uint16");
    case TILEDB_UINT32:
      return py::dtype("uint32");
    case TILEDB_UINT64:
      return py::dtype("uint64");
    case TILEDB_STRING_ASCII:
      return py::dtype("S1");
    case TILEDB_STRING_UTF8:
      return py::dtype("U1");
    case TILEDB_STRING_UTF16:
    case TILEDB_STRING_UTF32:
      TPY_ERROR_LOC("Unimplemented UTF16 or UTF32 string conversion!");
    case TILEDB_STRING_UCS2:
    case TILEDB_STRING_UCS4:
      TPY_ERROR_LOC("Unimplemented UCS2 or UCS4 string conversion!");
    case TILEDB_CHAR:
      return py::dtype("S1");
    case TILEDB_DATETIME_YEAR:
      return py::dtype("M8[Y]");
    case TILEDB_DATETIME_MONTH:
      return py::dtype("M8[M]");
    case TILEDB_DATETIME_WEEK:
      return py::dtype("M8[W]");
    case TILEDB_DATETIME_DAY:
      return py::dtype("M8[D]");
    case TILEDB_DATETIME_HR:
      return py::dtype("M8[h]");
    case TILEDB_DATETIME_MIN:
      return py::dtype("M8[m]");
    case TILEDB_DATETIME_SEC:
      return py::dtype("M8[s]");
    case TILEDB_DATETIME_MS:
      return py::dtype("M8[ms]");
    case TILEDB_DATETIME_US:
      return py::dtype("M8[us]");
    case TILEDB_DATETIME_NS:
      return py::dtype("M8[ns]");
    case TILEDB_DATETIME_PS:
      return py::dtype("M8[ps]");
    case TILEDB_DATETIME_FS:
      return py::dtype("M8[fs]");
    case TILEDB_DATETIME_AS:
      return py::dtype("M8[as]");

#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 3
    /* duration types map to timedelta */
    case TILEDB_TIME_HR:
      return py::dtype("m8[h]");
    case TILEDB_TIME_MIN:
      return py::dtype("m8[m]");
    case TILEDB_TIME_SEC:
      return py::dtype("m8[s]");
    case TILEDB_TIME_MS:
      return py::dtype("m8[ms]");
    case TILEDB_TIME_US:
      return py::dtype("m8[us]");
    case TILEDB_TIME_NS:
      return py::dtype("m8[ns]");
    case TILEDB_TIME_PS:
      return py::dtype("m8[ps]");
    case TILEDB_TIME_FS:
      return py::dtype("m8[fs]");
    case TILEDB_TIME_AS:
      return py::dtype("m8[as]");
#endif
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 7
    case TILEDB_BLOB:
      return py::dtype("bytes");
#endif

    case TILEDB_ANY:
      break;
    }
  } else if (cell_val_num == 2 && type == TILEDB_FLOAT32) {
    return py::dtype("complex64");
  } else if (cell_val_num == 2 && type == TILEDB_FLOAT64) {
    return py::dtype("complex128");
  } else if (type == TILEDB_CHAR || type == TILEDB_STRING_UTF8 ||
             type == TILEDB_STRING_ASCII) {
    std::string base_str;
    switch (type) {
    case TILEDB_CHAR:
    case TILEDB_STRING_ASCII:
      base_str = "|S";
      break;
    case TILEDB_STRING_UTF8:
      base_str = "|U";
      break;
    default:
      TPY_ERROR_LOC("internal error: unhandled string type");
    }
    if (cell_val_num < TILEDB_VAR_NUM) {
      base_str = base_str + std::to_string(cell_val_num);
    }
    return py::dtype(base_str);
  } else if (cell_val_num == TILEDB_VAR_NUM) {
    return tiledb_dtype(type, 1);
  } else if (cell_val_num > 1) {
    py::dtype base_dtype = tiledb_dtype(type, 1);
    py::tuple rec_elem = py::make_tuple("", base_dtype);
    py::list rec_list;
    for (size_t i = 0; i < cell_val_num; i++)
      rec_list.append(rec_elem);
    auto np = py::module::import("numpy");
    // note: we call the 'dtype' constructor b/c py::dtype does not accept list
    auto np_dtype = np.attr("dtype");
    return np_dtype(rec_list);
  }

  TPY_ERROR_LOC("tiledb datatype not understood ('" +
                tiledb::impl::type_to_str(type) +
                "', cell_val_num: " + std::to_string(cell_val_num) + ")");
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
