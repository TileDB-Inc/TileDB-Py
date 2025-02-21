#include <nanobind/nanobind.h>

namespace tiledbpy {

namespace nb = nanobind;

void init_core(nb::module_&);
// void _debug(nb::module_ &);
void init_fragment(nb::module_&);
// void init_query_condition(nb::module_ &);
void init_schema_evolution(nb::module_&);
#if defined(TILEDB_SERIALIZATION)
void init_serialization(nb::module_&);
void init_test_serialization(nb::module_&);
#endif
void init_test_metadata(nb::module_&);
void init_test_webp_filter(nb::module_&);

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
