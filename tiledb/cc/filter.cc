#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

class PyFilter : public Filter {
public:
  PyFilter(const Context &ctx, tiledb_filter_type_t type) : Filter(ctx, type){};
  PyFilter(Filter &filter) : Filter(filter){};

  virtual std::map<std::string, int> attrs() const {
    return std::map<std::string, int>();
  };

  virtual std::string name() { return "PyFilter"; };

  std::string repr() {
    std::stringstream ss;
    ss << this->name() << "(";
    for (auto a : this->attrs()) {
      ss << a.first << "=" << std::to_string(a.second);
    }
    ss << ")";
    return ss.str();
  };

  std::string _repr_html_() {
    std::stringstream ss;
    return ss.str();
  };
};

inline bool operator==(const PyFilter &lhs, const PyFilter &rhs) {
  py::object pylhs = py::cast(lhs);
  py::object pyrhs = py::cast(rhs);

  for (auto a : lhs.attrs()) {
    if (py::int_(pylhs.attr(a.first.c_str())) !=
        py::int_(pyrhs.attr(a.first.c_str())))
      return false;
  }
  return true;
}

inline bool operator!=(const PyFilter &lhs, const PyFilter &rhs) {
  return !operator==(lhs, rhs);
}

class NoOpFilter : public PyFilter {
public:
  NoOpFilter(const Context &ctx) : PyFilter(ctx, TILEDB_FILTER_NONE){};
  NoOpFilter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "NoOpFilter"; }
};

class CompressionFilter : public PyFilter {
public:
  CompressionFilter(int32_t level, tiledb_filter_type_t type,
                    const Context &ctx)
      : PyFilter(ctx, type) {
    this->set_option(TILEDB_COMPRESSION_LEVEL, level);
  };
  CompressionFilter(Filter &filter) : PyFilter(filter){};

  int32_t level() {
    int32_t level;
    this->get_option<int32_t>(TILEDB_COMPRESSION_LEVEL, &level);
    return level;
  }
};

class GzipFilter : public CompressionFilter {
public:
  GzipFilter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_GZIP, ctx){};
  GzipFilter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "GzipFilter"; }
};

class ZstdFilter : public CompressionFilter {
public:
  ZstdFilter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_ZSTD, ctx){};
  ZstdFilter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "ZstdFilter"; }
};
class LZ4Filter : public CompressionFilter {
public:
  LZ4Filter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_LZ4, ctx){};
  LZ4Filter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "LZ4Filter"; }
};

class Bzip2Filter : public CompressionFilter {
public:
  Bzip2Filter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_BZIP2, ctx){};
  Bzip2Filter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "Bzip2Filter"; }
};

class RleFilter : public CompressionFilter {
public:
  RleFilter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_LZ4, ctx){};
  RleFilter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "RleFilter"; }
};

class DoubleDeltaFilter : public CompressionFilter {
public:
  DoubleDeltaFilter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_LZ4, ctx){};
  DoubleDeltaFilter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "DoubleDeltaFilter"; }
};

class DictionaryFilter : public CompressionFilter {
public:
  DictionaryFilter(int32_t level, const Context &ctx)
      : CompressionFilter(level, TILEDB_FILTER_DICTIONARY, ctx){};
  DictionaryFilter(Filter &filter) : CompressionFilter(filter){};

  std::string name() { return "DictionaryFilter"; }
};

class BitShuffleFilter : public PyFilter {
public:
  BitShuffleFilter(const Context &ctx)
      : PyFilter(ctx, TILEDB_FILTER_BYTESHUFFLE){};
  BitShuffleFilter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "BitShuffleFilter"; }
};

class ByteShuffleFilter : public PyFilter {
public:
  ByteShuffleFilter(const Context &ctx)
      : PyFilter(ctx, TILEDB_FILTER_BITSHUFFLE){};
  ByteShuffleFilter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "ByteShuffleFilter"; }
};

class BitWidthReductionFilter : public PyFilter {
public:
  BitWidthReductionFilter(uint32_t window, const Context &ctx)
      : PyFilter(ctx, TILEDB_FILTER_BIT_WIDTH_REDUCTION) {
    this->set_option<uint32_t>(TILEDB_BIT_WIDTH_MAX_WINDOW, window);
  };
  BitWidthReductionFilter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "BitWidthReductionFilter"; }

  uint32_t window() {
    uint32_t window;
    this->get_option<uint32_t>(TILEDB_BIT_WIDTH_MAX_WINDOW, &window);
    return window;
  }
};

class PositiveDeltaFilter : public PyFilter {
public:
  PositiveDeltaFilter(uint32_t window, const Context &ctx)
      : PyFilter(ctx, TILEDB_FILTER_POSITIVE_DELTA) {
    this->set_option<uint32_t>(TILEDB_POSITIVE_DELTA_MAX_WINDOW, window);
  };
  PositiveDeltaFilter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "PositiveDeltaFilter"; }

  uint32_t window() {
    uint32_t window;
    this->get_option<uint32_t>(TILEDB_POSITIVE_DELTA_MAX_WINDOW, &window);
    return window;
  }
};

class ChecksumMD5Filter : public PyFilter {
public:
  ChecksumMD5Filter(const Context &ctx)
      : PyFilter(ctx, TILEDB_FILTER_CHECKSUM_MD5){};
  ChecksumMD5Filter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "ChecksumMD5Filter"; }
};

class ChecksumSHA256Filter : public PyFilter {
public:
  ChecksumSHA256Filter(const Context &ctx)
      : PyFilter(ctx, TILEDB_FILTER_CHECKSUM_SHA256){};
  ChecksumSHA256Filter(Filter &filter) : PyFilter(filter){};

  std::string name() { return "ChecksumSHA256Filter"; }
};

py::object pyfilter_to_pyobject(PyFilter pyfilter) {
  switch (pyfilter.filter_type()) {
  case TILEDB_FILTER_NONE:
    return py::cast(NoOpFilter(pyfilter));
  case TILEDB_FILTER_GZIP:
    return py::cast(GzipFilter(pyfilter));
  case TILEDB_FILTER_ZSTD:
    return py::cast(ZstdFilter(pyfilter));
  case TILEDB_FILTER_LZ4:
    return py::cast(LZ4Filter(pyfilter));
  case TILEDB_FILTER_RLE:
    return py::cast(RleFilter(pyfilter));
  case TILEDB_FILTER_BZIP2:
    return py::cast(Bzip2Filter(pyfilter));
  case TILEDB_FILTER_DOUBLE_DELTA:
    return py::cast(DoubleDeltaFilter(pyfilter));
  case TILEDB_FILTER_BIT_WIDTH_REDUCTION:
    return py::cast(BitWidthReductionFilter(pyfilter));
  case TILEDB_FILTER_BITSHUFFLE:
    return py::cast(BitShuffleFilter(pyfilter));
  case TILEDB_FILTER_BYTESHUFFLE:
    return py::cast(ByteShuffleFilter(pyfilter));
  case TILEDB_FILTER_POSITIVE_DELTA:
    return py::cast(PositiveDeltaFilter(pyfilter));
  case TILEDB_FILTER_CHECKSUM_MD5:
    return py::cast(ChecksumMD5Filter(pyfilter));
  case TILEDB_FILTER_CHECKSUM_SHA256:
    return py::cast(ChecksumSHA256Filter(pyfilter));
  case TILEDB_FILTER_DICTIONARY:
    return py::cast(DictionaryFilter(pyfilter));
  default:
    throw py::type_error("pyfilter_to_pyobject given invalid filter type");
  }
}

void init_filter(py::module &m) {
  py::class_<PyFilter>(m, "Filter");

  py::class_<NoOpFilter, PyFilter>(m, "NoOpFilter")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init([]() { return NoOpFilter(Context()); }))
      .def("__repr__", &NoOpFilter::repr)
      .def(py::self == py::self);

  py::class_<GzipFilter, PyFilter>(m, "GzipFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &GzipFilter::level)
      .def("__repr__", &GzipFilter::repr)
      .def(py::self == py::self);

  py::class_<ZstdFilter, PyFilter>(m, "ZstdFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &ZstdFilter::level)
      .def("__repr__", &ZstdFilter::repr)
      .def(py::self == py::self);

  py::class_<LZ4Filter, PyFilter>(m, "LZ4Filter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &LZ4Filter::level)
      .def("__repr__", &LZ4Filter::repr)
      .def(py::self == py::self);

  py::class_<Bzip2Filter, PyFilter>(m, "Bzip2Filter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &Bzip2Filter::level)
      .def("__repr__", &Bzip2Filter::repr)
      .def(py::self == py::self);

  py::class_<RleFilter, PyFilter>(m, "RleFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &RleFilter::level)
      .def("__repr__", &RleFilter::repr)
      .def(py::self == py::self);

  py::class_<DoubleDeltaFilter, PyFilter>(m, "DoubleDeltaFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &DoubleDeltaFilter::level)
      .def("__repr__", &DoubleDeltaFilter::repr)
      .def(py::self == py::self);

  py::class_<DictionaryFilter, PyFilter>(m, "DictionaryFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("level") = -1, py::arg("ctx") = Context())
      .def_property_readonly("level", &DictionaryFilter::level)
      .def("__repr__", &DictionaryFilter::repr)
      .def(py::self == py::self);

  py::class_<BitShuffleFilter, PyFilter>(m, "BitShuffleFilter")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init([]() { return BitShuffleFilter(Context()); }))
      .def("__repr__", &BitShuffleFilter::repr)
      .def(py::self == py::self);

  py::class_<ByteShuffleFilter, PyFilter>(m, "ByteShuffleFilter")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init([]() { return ByteShuffleFilter(Context()); }))
      .def("__repr__", &ByteShuffleFilter::repr)
      .def(py::self == py::self);

  py::class_<BitWidthReductionFilter, PyFilter>(m, "BitWidthReductionFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("window") = -1, py::arg("ctx") = Context())
      .def_property_readonly("window", &BitWidthReductionFilter::window)
      .def("__repr__", &BitWidthReductionFilter::repr)
      .def(py::self == py::self);

  py::class_<PositiveDeltaFilter, PyFilter>(m, "PositiveDeltaFilter")
      .def(py::init<int, const Context &>(), py::keep_alive<1, 3>(),
           py::arg("window") = -1, py::arg("ctx") = Context())
      .def_property_readonly("window", &PositiveDeltaFilter::window)
      .def("__repr__", &PositiveDeltaFilter::repr)
      .def(py::self == py::self);

  py::class_<ChecksumMD5Filter, PyFilter>(m, "ChecksumMD5Filter")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init([]() { return ChecksumMD5Filter(Context()); }))
      .def("__repr__", &ChecksumMD5Filter::repr)
      .def(py::self == py::self);

  py::class_<ChecksumSHA256Filter, PyFilter>(m, "ChecksumSHA256Filter")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init([]() { return ChecksumSHA256Filter(Context()); }))
      .def("__repr__", &ChecksumSHA256Filter::repr)
      .def(py::self == py::self);

  py::class_<FilterList>(m, "FilterList")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init<const Context &, py::capsule>(), py::keep_alive<1, 2>())

      .def(py::init([](py::object filters, py::object chunksize,
                       const Context &ctx) {
             py::list _filters;
             uint32_t _chunksize;
             //  Context _ctx;

             if (py::isinstance<py::none>(filters)) {
               _filters = py::list();
             } else {
               _filters = py::list(filters);
             }

             if (py::isinstance<py::none>(chunksize)) {
               _chunksize = 65536;
             } else {
               _chunksize = py::int_(chunksize);
             }

             //  if (py::isinstance<py::none>(ctx)) {
             //    _ctx = Context();
             //  } else {
             //    _ctx = ctx.cast<Context>();
             //  }

             FilterList filter_list = FilterList(ctx);

             for (auto filter : _filters) {
               try {
                 filter_list.add_filter(filter.cast<PyFilter>());
               } catch (...) {
                 throw py::type_error(
                     "filters argument must be an iterable of TileDB "
                     "filter objects");
               }
             }

             filter_list.set_max_chunk_size(_chunksize);

             return filter_list;
           }),
           py::keep_alive<1, 4>(), py::arg("filters") = py::none(),
           py::arg("chunksize") = py::none(), py::arg("ctx") = Context())

      .def("__capsule__",
           [](FilterList &filterlist) {
             return py::capsule(filterlist.ptr().get(), "fl", nullptr);
           })

      .def("_repr_html_", [](FilterList &filterlist) { return ""; })

      .def_property("chunksize", &FilterList::max_chunk_size,
                    &FilterList::set_max_chunk_size)

      .def("__len__", &FilterList::nfilters)

      .def("__eq__",
           [](FilterList &self, py::object other) {
             if (other.is_none())
               return false;
             if (self.nfilters() != py::len(other))
               return false;

             auto other_it = py::iter(py::iterable(other));
             for (uint32_t i = 0; i < self.nfilters(); ++i) {
               if (py::isinstance<PyFilter>(*other_it) == true) {
                 Filter f = self.filter(i);
                 if (PyFilter(f) != other_it->cast<PyFilter>()) {
                   return false;
                 }
               }
               ++other_it;
             }
             return true;
           })

      .def("__getitem__",
           [](FilterList &filter_list, uint32_t filter_index) {
             if (filter_list.nfilters() == 0 or filter_index < 0 or
                 filter_index > (filter_list.nfilters() - 1)) {
               throw std::out_of_range("FilterList index out of range");
             }

             auto filter = filter_list.filter(filter_index);
             return pyfilter_to_pyobject(PyFilter(filter));
           })

      .def("__getitem__",
           [](FilterList &filter_list, py::slice filter_indexes) -> py::list {
             py::list output;
             py::tuple slice = py::make_tuple(filter_indexes);
             uint32_t start = py::int_(slice[0]);
             uint32_t stop = py::int_(slice[1]);
             uint32_t step = py::int_(slice[2]);

             for (uint32_t i = start; i < stop; ++step) {
               Filter filter = filter_list.filter(i);
               output.append(pyfilter_to_pyobject(PyFilter(filter)));
             }

             return output;
           })

      .def(
          "append",
          [](FilterList &filterlist, PyFilter &filter) {
            filterlist.add_filter(filter);
          },
          py::keep_alive<1, 2>())

      .def("append", &FilterList::add_filter, py::keep_alive<1, 2>());
}

} // namespace libtiledbcpp
