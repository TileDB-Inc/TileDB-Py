#include <pybind11/pybind11.h>

namespace tiledbpy {

namespace nb = nanobind;

void init_core(nb::module&);
// void _debug(nb::module &);
void init_fragment(nb::module&);
// void init_query_condition(nb::module &);
void init_schema_evolution(nb::module&);
#if defined(TILEDB_SERIALIZATION)
void init_serialization(nb::module&);
void init_test_serialization(nb::module&);
#endif
void init_test_metadata(nb::module&);
void init_test_webp_filter(nb::module&);

NB_MODULE(main, m) {
    init_core(m);
    //_debug(m);
    init_fragment(m);
    //_query_condition(m);
    init_schema_evolution(m);
#if defined(TILEDB_SERIALIZATION)
    init_serialization(m);
    init_test_serialization(m);
#endif
    init_test_metadata(m);
    init_test_webp_filter(m);
}

}  // namespace tiledbpy
