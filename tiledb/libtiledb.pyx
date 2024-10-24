#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.version cimport PY_MAJOR_VERSION

include "common.pxi"
include "indexing.pyx"
import collections.abc
from json import loads as json_loads

from .cc import TileDBError
from .ctx import Config, Ctx, default_ctx
from .domain_indexer import DomainIndexer
from .vfs import VFS
from .sparse_array import SparseArrayImpl
from .dense_array import DenseArrayImpl
from .array import Array

###############################################################################
#     Numpy initialization code (critical)                                    #
###############################################################################

# https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array
np.import_array()

###############################################################################
#    Utility/setup                                                            #
###############################################################################

# Use unified numpy printing
np.set_printoptions(legacy="1.21" if np.lib.NumpyVersion(np.__version__) >= "1.22.0" else False)


cdef tiledb_ctx_t* safe_ctx_ptr(object ctx):
    if ctx is None:
        raise TileDBError("internal error: invalid Ctx object")
    return <tiledb_ctx_t*>PyCapsule_GetPointer(ctx.__capsule__(), "ctx")

def version():
    """Return the version of the linked ``libtiledb`` shared library

    :rtype: tuple
    :return: Semver version (major, minor, rev)

    """
    cdef:
        int major = 0
        int minor = 0
        int rev = 0
    tiledb_version(&major, &minor, &rev)
    return major, minor, rev


# note: this function is cdef, so it must return a python object in order to
#       properly forward python exceptions raised within the function. See:
#       https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#error-return-values
cdef dict get_query_fragment_info(tiledb_ctx_t* ctx_ptr,
                                   tiledb_query_t* query_ptr):

    cdef int rc = TILEDB_OK
    cdef uint32_t num_fragments
    cdef Py_ssize_t fragment_idx
    cdef const char* fragment_uri_ptr
    cdef unicode fragment_uri
    cdef uint64_t fragment_t1, fragment_t2
    cdef dict result = dict()

    rc = tiledb_query_get_fragment_num(ctx_ptr, query_ptr, &num_fragments)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    if (num_fragments < 1):
        return result

    for fragment_idx in range(0, num_fragments):

        rc = tiledb_query_get_fragment_uri(ctx_ptr, query_ptr, fragment_idx, &fragment_uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_get_fragment_timestamp_range(
                ctx_ptr, query_ptr, fragment_idx, &fragment_t1, &fragment_t2)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        fragment_uri = fragment_uri_ptr.decode('UTF-8')
        result[fragment_uri] = (fragment_t1, fragment_t2)

    return result

def _write_array_wrapper(
        object tiledb_array,
        object subarray,
        list coordinates,
        list buffer_names,
        list values,
        dict labels,
        dict nullmaps,
        bint issparse,
    ):

    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(tiledb_array.ctx)
    cdef tiledb_array_t* array_ptr = <tiledb_array_t*>PyCapsule_GetPointer(tiledb_array.array.__capsule__(), "array")
    cdef dict fragment_info = tiledb_array.last_fragment_info
    _write_array(ctx_ptr, array_ptr, tiledb_array, subarray, coordinates, buffer_names, values, labels, nullmaps, fragment_info, issparse)

cdef _write_array(
        tiledb_ctx_t* ctx_ptr,
        tiledb_array_t* array_ptr,
        object tiledb_array,
        object subarray,
        list coordinates,
        list buffer_names,
        list values,
        dict labels,
        dict nullmaps,
        dict fragment_info,
        bint issparse,
    ):

    # used for buffer conversion (local import to avoid circularity)
    from .main import array_to_buffer

    cdef bint isfortran = False
    cdef Py_ssize_t nattr = len(buffer_names)
    cdef Py_ssize_t nlabel = len(labels)

    # Create arrays to hold buffer sizes
    cdef Py_ssize_t nbuffer = nattr + nlabel
    if issparse:
        nbuffer += tiledb_array.schema.ndim
    cdef np.ndarray buffer_sizes = np.zeros((nbuffer,), dtype=np.uint64)
    cdef np.ndarray buffer_offsets_sizes = np.zeros((nbuffer,),  dtype=np.uint64)
    cdef np.ndarray nullmaps_sizes = np.zeros((nbuffer,), dtype=np.uint64)

    # Create lists for data and offset buffers
    output_values = list()
    output_offsets = list()

    # Set data and offset buffers for attributes
    for i in range(nattr):
        # if dtype is ASCII, ensure all characters are valid
        if tiledb_array.schema.attr(i).isascii:
            try:
                values[i] = np.asarray(values[i], dtype=np.bytes_)
            except Exception as exc:
                raise TileDBError(f'dtype of attr {tiledb_array.schema.attr(i).name} is "ascii" but attr_val contains invalid ASCII characters')

        attr = tiledb_array.schema.attr(i)

        if attr.isvar:
            try:
                if attr.isnullable:
                    if(np.issubdtype(attr.dtype, np.str_) 
                        or np.issubdtype(attr.dtype, np.bytes_)):
                        attr_val = np.array(["" if v is None else v for v in values[i]])
                    else:
                        attr_val = np.nan_to_num(values[i])
                else:
                    attr_val = values[i]
                buffer, offsets = array_to_buffer(attr_val, True, False)
            except Exception as exc:
                raise type(exc)(f"Failed to convert buffer for attribute: '{attr.name}'") from exc
            buffer_offsets_sizes[i] = offsets.nbytes
        else:
            buffer, offsets = values[i], None

        buffer_sizes[i] = buffer.nbytes
        output_values.append(buffer)
        output_offsets.append(offsets)

    # Check value layouts
    if len(values) and nattr > 1:
        value = output_values[0]
        isfortran = value.ndim > 1 and value.flags.f_contiguous
        for value in values:
            if value.ndim > 1 and value.flags.f_contiguous and not isfortran:
                raise ValueError("mixed C and Fortran array layouts")

    # Set data and offsets buffers for dimensions (sparse arrays only)
    ibuffer = nattr
    if issparse:
        for dim_idx, coords in enumerate(coordinates):
            if tiledb_array.schema.domain.dim(dim_idx).isvar:
                buffer, offsets = array_to_buffer(coords, True, False)
                buffer_sizes[ibuffer] = buffer.nbytes
                buffer_offsets_sizes[ibuffer] = offsets.nbytes
            else:
                buffer, offsets = coords, None
                buffer_sizes[ibuffer] = buffer.nbytes
            output_values.append(buffer)
            output_offsets.append(offsets)

            name = tiledb_array.schema.domain.dim(dim_idx).name
            buffer_names.append(name)

            ibuffer = ibuffer + 1

    for label_name, label_values in labels.items():
        # Append buffer name
        buffer_names.append(label_name)
        # Get label data buffer and offsets buffer for the labels
        dim_label = tiledb_array.schema.dim_label(label_name)
        if dim_label.isvar:
            buffer, offsets = array_to_buffer(label_values, True, False)
            buffer_sizes[ibuffer] = buffer.nbytes
            buffer_offsets_sizes[ibuffer] = offsets.nbytes
        else:
            buffer, offsets = label_values, None
            buffer_sizes[ibuffer] = buffer.nbytes
        # Append the buffers
        output_values.append(buffer)
        output_offsets.append(offsets)

        ibuffer = ibuffer + 1


    # Allocate the query
    cdef int rc = TILEDB_OK
    cdef tiledb_query_t* query_ptr = NULL
    rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    # Set layout
    cdef tiledb_layout_t layout = (
            TILEDB_UNORDERED
            if issparse
            else (TILEDB_COL_MAJOR if isfortran else TILEDB_ROW_MAJOR)
    )
    rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)

    # Create and set the subarray for the query (dense arrays only)
    cdef np.ndarray s_start
    cdef np.ndarray s_end
    cdef np.dtype dim_dtype = None
    cdef void* s_start_ptr = NULL
    cdef void* s_end_ptr = NULL
    cdef tiledb_subarray_t* subarray_ptr = NULL
    if not issparse:
        subarray_ptr = <tiledb_subarray_t*>PyCapsule_GetPointer(
                subarray.__capsule__(), "subarray")
        # Set the subarray on the query
        rc = tiledb_query_set_subarray_t(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

    # Set buffers on the query
    cdef bytes bname
    cdef void* buffer_ptr = NULL
    cdef uint64_t* offsets_buffer_ptr = NULL
    cdef uint8_t* nulmap_buffer_ptr = NULL
    cdef uint64_t* buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_sizes)
    cdef uint64_t* offsets_buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_offsets_sizes)
    cdef uint64_t* nullmaps_sizes_ptr = <uint64_t*> np.PyArray_DATA(nullmaps_sizes)
    try:
        for i, buffer_name in enumerate(buffer_names):
            # Get utf-8 version of the name for C-API calls
            bname = buffer_name.encode('UTF-8')

            # Set data buffer
            buffer_ptr = np.PyArray_DATA(output_values[i])
            rc = tiledb_query_set_data_buffer(
                    ctx_ptr, query_ptr, bname, buffer_ptr, &(buffer_sizes_ptr[i]))
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

            # Set offsets buffer
            if output_offsets[i] is not None:
                offsets_buffer_ptr = <uint64_t*>np.PyArray_DATA(output_offsets[i])
                rc = tiledb_query_set_offsets_buffer(
                        ctx_ptr,
                        query_ptr,
                        bname,
                        offsets_buffer_ptr,
                        &(offsets_buffer_sizes_ptr[i])
                )
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

            # Set validity buffer
            if buffer_name in nullmaps:
                # NOTE: validity map is owned *by the caller*
                nulmap = nullmaps[buffer_name]
                nullmaps_sizes[i] = len(nulmap)
                nulmap_buffer_ptr = <uint8_t*>np.PyArray_DATA(nulmap)
                rc = tiledb_query_set_validity_buffer(
                    ctx_ptr,
                    query_ptr,
                    bname,
                    nulmap_buffer_ptr,
                    &(nullmaps_sizes_ptr[i])
                )
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_finalize(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        if fragment_info is not False:
            assert(type(fragment_info) is dict)
            fragment_info.clear()
            fragment_info.update(get_query_fragment_info(ctx_ptr, query_ptr))

    finally:
        tiledb_query_free(&query_ptr)
    return

cdef _raise_tiledb_error(tiledb_error_t* err_ptr):
    cdef const char* err_msg_ptr = NULL
    ret = tiledb_error_message(err_ptr, &err_msg_ptr)
    if ret != TILEDB_OK:
        tiledb_error_free(&err_ptr)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error message")
    cdef unicode message_string
    try:
        message_string = err_msg_ptr.decode('UTF-8', 'strict')
    finally:
        tiledb_error_free(&err_ptr)
    raise TileDBError(message_string)


cdef _raise_ctx_err(tiledb_ctx_t* ctx_ptr, int rc):
    if rc == TILEDB_OK:
        return
    if rc == TILEDB_OOM:
        raise MemoryError()
    cdef tiledb_error_t* err_ptr = NULL
    cdef int ret = tiledb_ctx_get_last_error(ctx_ptr, &err_ptr)
    if ret != TILEDB_OK:
        tiledb_error_free(&err_ptr)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error object from ctx")
    _raise_tiledb_error(err_ptr)


cpdef unicode ustring(object s):
    """Coerce a python object to a unicode string"""

    if type(s) is unicode:
        return <unicode> s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        return (<bytes> s).decode('ascii')
    elif isinstance(s, unicode):
        return unicode(s)
    raise TypeError(
        "ustring() must be a string or a bytes-like object"
        ", not {0!r}".format(type(s)))


cdef bytes unicode_path(object path):
    """Returns a UTF-8 encoded byte representation of a given URI path string"""
    return ustring(path).encode('UTF-8')


###############################################################################
#                                                                             #
#    CLASS DEFINITIONS                                                        #
#                                                                             #
###############################################################################


cdef class Query(object):
    """
    Proxy object returned by query() to index into original array
    on a subselection of attribute in a defined layout order

    See documentation of Array.query
    """

    def __init__(self, array, attrs=None, cond=None, dims=None,
                 coords=False, index_col=True, order=None,
                 use_arrow=None, return_arrow=False, return_incomplete=False):
        if array.mode not in  ('r', 'd'):
            raise ValueError("array mode must be read or delete mode")

        if dims is not None and coords == True:
            raise ValueError("Cannot pass both dims and coords=True to Query")

        cdef list dims_to_set = list()

        if dims is False:
            self.dims = False
        elif dims != None and dims != True:
            domain = array.schema.domain
            for dname in dims:
                if not domain.has_dim(dname):
                    raise TileDBError(f"Selected dimension does not exist: '{dname}'")
            self.dims = [unicode(dname) for dname in dims]
        elif coords == True or dims == True:
            domain = array.schema.domain
            self.dims = [domain.dim(i).name for i in range(domain.ndim)]

        if attrs is not None:
            for name in attrs:
                if not array.schema.has_attr(name):
                    raise TileDBError(f"Selected attribute does not exist: '{name}'")
        self.attrs = attrs
        self.cond = cond

        if order == None:
            if array.schema.sparse:
                self.order = 'U' # unordered
            else:
                self.order = 'C' # row-major
        else:
            self.order = order

        # reference to the array we are querying
        self.array = array
        self.coords = coords
        self.index_col = index_col
        self.return_arrow = return_arrow
        if return_arrow:
            if use_arrow is None:
                use_arrow = True
            if not use_arrow:
                raise TileDBError("Cannot initialize return_arrow with use_arrow=False")
        self.use_arrow = use_arrow

        if return_incomplete and not array.schema.sparse:
            raise TileDBError("Incomplete queries are only supported for sparse arrays at this time")

        self.return_incomplete = return_incomplete

        self.domain_index = DomainIndexer(array, query=self)

    def __getitem__(self, object selection):
        if self.return_arrow:
            raise TileDBError("`return_arrow=True` requires .df indexer`")

        return self.array.subarray(selection,
                                attrs=self.attrs,
                                cond=self.cond,
                                coords=self.coords if self.coords else self.dims,
                                order=self.order)
    
    def agg(self, aggs):
        """
        Calculate an aggregate operation for a given attribute. Available 
        operations are sum, min, max, mean, count, and null_count (for nullable
        attributes only). Aggregates may be combined with other query operations 
        such as query conditions and slicing.

        The input may be a single operation, a list of operations, or a 
        dictionary with attribute mapping to a single operation or list of 
        operations.

        For undefined operations on max and min, which can occur when a nullable
        attribute contains only nulled data at the given coordinates or when 
        there is no data read for the given query (e.g. query conditions that do
        not match any values or coordinates that contain no data)), invalid
        results are represented as np.nan for attributes of floating point types
        and None for integer types.

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.from_numpy(path, np.arange(1, 10)) as A:
        ...     pass

        >>> # Note that tiledb.from_numpy creates anonymous attributes, so the
        >>> # name of the attribute is represented as an empty string

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query().agg("sum")[:]
        45

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query(cond="attr('') < 5").agg(["count", "mean"])[:]
        {'count': 9, 'mean': 2.5}

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query().agg({"": ["max", "min"]})[2:7]
        {'max': 7, 'min': 3}

        :param agg: The input attributes and operations to apply aggregations on
        :returns: single value for single operation on one attribute, a dictionary
            of attribute keys associated with a single value for a single operation
            across multiple attributes, or a dictionary of attribute keys that maps
            to a dictionary of operation labels with the associated value
        """
        schema = self.array.schema
        attr_to_aggs_map = {}
        if isinstance(aggs, dict):
            attr_to_aggs_map = {
                a: (
                    tuple([aggs[a]]) 
                    if isinstance(aggs[a], str) 
                    else tuple(aggs[a])
                )
                for a in aggs
            }
        elif isinstance(aggs, str):
            attrs = tuple(schema.attr(i).name for i in range(schema.nattr))
            attr_to_aggs_map = {a: (aggs,) for a in attrs}
        elif isinstance(aggs, collections.abc.Sequence):
            attrs = tuple(schema.attr(i).name for i in range(schema.nattr))
            attr_to_aggs_map = {a: tuple(aggs) for a in attrs}

        from .aggregation import Aggregation
        return Aggregation(self, attr_to_aggs_map)

    @property
    def array(self):
        return self.array

    @property
    def attrs(self):
        """List of attributes to include in Query."""
        return self.attrs

    @property
    def cond(self):
        """QueryCondition used to filter attributes or dimensions in Query."""
        return self.cond

    @property
    def dims(self):
        """List of dimensions to include in Query."""
        return self.dims

    @property
    def coords(self):
        """
        True if query should include (return) coordinate values.

        :rtype: bool
        """
        return self.coords

    @property
    def order(self):
        """Return underlying Array order."""
        return self.order

    @property
    def index_col(self):
        """List of columns to set as index for dataframe queries, or None."""
        return self.index_col

    @property
    def use_arrow(self):
        return self.use_arrow

    @property
    def return_arrow(self):
        return self.return_arrow

    @property
    def return_incomplete(self):
        return self.return_incomplete

    @property
    def domain_index(self):
        """Apply Array.domain_index with query parameters."""
        return self.domain_index

    def label_index(self, labels):
        """Apply Array.label_index with query parameters."""
        from .multirange_indexing import LabelIndexer
        return LabelIndexer(self.array, tuple(labels), query=self)

    @property
    def multi_index(self):
        """Apply Array.multi_index with query parameters."""
        # Delayed to avoid circular import
        from .multirange_indexing import MultiRangeIndexer
        return MultiRangeIndexer(self.array, query=self)

    @property
    def df(self):
        """Apply Array.multi_index with query parameters and return result
           as a Pandas dataframe."""
        # Delayed to avoid circular import
        from .multirange_indexing import DataFrameIndexer
        return DataFrameIndexer(self.array, query=self, use_arrow=self.use_arrow)

    def get_stats(self, print_out=True, json=False):
        """Retrieves the stats from a TileDB query.

        :param print_out: Print string to console (default True), or return as string
        :param json: Return stats JSON object (default: False)
        """
        pyquery = self.array.pyquery
        if pyquery is None:
            return ""
        stats = self.array.pyquery.get_stats()
        if json:
            stats = json_loads(stats)
        if print_out:
            print(stats)
        else:
            return stats

    def submit(self):
        """An alias for calling the regular indexer [:]"""
        return self[:]

def write_direct_dense(self: Array, np.ndarray array not None, **kw):
    """
    Write directly to given array attribute with minimal checks,
    assumes that the numpy array is the same shape as the array's domain

    :param np.ndarray array: Numpy contiguous dense array of the same dtype \
        and shape and layout of the DenseArray instance
    :raises ValueError: array is not contiguous
    :raises: :py:exc:`tiledb.TileDBError`

    """
    append_dim = kw.pop("append_dim", None)
    mode = kw.pop("mode", "ingest")
    start_idx = kw.pop("start_idx", None)

    if not self.isopen or self.mode != 'w':
        raise TileDBError("DenseArray is not opened for writing")
    if self.schema.nattr != 1:
        raise ValueError("cannot write_direct to a multi-attribute DenseArray")
    if not array.flags.c_contiguous and not array.flags.f_contiguous:
        raise ValueError("array is not contiguous")

    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
    cdef tiledb_array_t* array_ptr = <tiledb_array_t*>PyCapsule_GetPointer(self.array.__capsule__(), "array")

    # attr name
    attr = self.schema.attr(0)
    cdef bytes battr_name = attr._internal_name.encode('UTF-8')
    cdef const char* attr_name_ptr = PyBytes_AS_STRING(battr_name)

    cdef void* buff_ptr = np.PyArray_DATA(array)
    cdef uint64_t buff_size = array.nbytes
    cdef np.ndarray subarray = np.zeros(2*array.ndim, np.uint64)

    try:
        use_global_order = self.ctx.config().get(
            "py.use_global_order_1d_write") == "true"
    except KeyError:
        use_global_order = False

    cdef tiledb_layout_t layout = TILEDB_ROW_MAJOR
    if array.ndim == 1 and use_global_order:
        layout = TILEDB_GLOBAL_ORDER
    elif array.flags.f_contiguous:
        layout = TILEDB_COL_MAJOR

    cdef tiledb_query_t* query_ptr = NULL
    cdef tiledb_subarray_t* subarray_ptr = NULL
    cdef int rc = TILEDB_OK
    rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)
    try:
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        range_start_idx = start_idx or 0
        for n in range(array.ndim):
            subarray[n*2] = range_start_idx
            subarray[n*2 + 1] = array.shape[n] + range_start_idx - 1

        if mode == "append":
            with Array.load_typed(self.uri) as A:
                ned = A.nonempty_domain()

            if array.ndim <= append_dim:
                raise IndexError("`append_dim` out of range")

            if array.ndim != len(ned):
                raise ValueError(
                    "The number of dimension of the TileDB array and "
                    "Numpy array to append do not match"
                )

            for n in range(array.ndim):
                if n == append_dim:
                    if start_idx is not None:
                        range_start_idx = start_idx
                        range_end_idx = array.shape[n] + start_idx -1
                    else:
                        range_start_idx = ned[n][1] + 1
                        range_end_idx = array.shape[n] + ned[n][1]

                    subarray[n*2] = range_start_idx
                    subarray[n*2 + 1] = range_end_idx
                else:
                    if array.shape[n] != ned[n][1] - ned[n][0] + 1:
                        raise ValueError(
                            "The input Numpy array must be of the same "
                            "shape as the TileDB array, exluding the "
                            "`append_dim`, but the Numpy array at index "
                            f"{n} has {array.shape[n]} dimension(s) and "
                            f"the TileDB array has {ned[n][1]-ned[n][0]}."
                        )

        rc = tiledb_subarray_alloc(ctx_ptr, array_ptr, &subarray_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_subarray_set_subarray(
                ctx_ptr,
                subarray_ptr,
                <void*>np.PyArray_DATA(subarray)
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_set_subarray_t(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_set_data_buffer(
                ctx_ptr,
                query_ptr,
                attr_name_ptr,
                buff_ptr,
                &buff_size
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_finalize(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
    finally:
        tiledb_subarray_free(&subarray_ptr)
        tiledb_query_free(&query_ptr)
    return
