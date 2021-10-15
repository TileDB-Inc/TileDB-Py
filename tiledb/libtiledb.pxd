from libc.stdio cimport FILE
from libc.stdint cimport uint64_t, uint32_t

IF TILEDBPY_MODULAR:
    from .indexing cimport DomainIndexer

include "common.pxi"

cdef extern from "tiledb/tiledb.h":
    # Constants
    enum: TILEDB_OK
    enum: TILEDB_ERR
    enum: TILEDB_OOM

    enum: TILEDB_VAR_NUM
    unsigned int tiledb_var_num()

    enum: TILEDB_COORDS
    const char* tiledb_coords()

    enum: TILEDB_MAX_PATH
    unsigned int tiledb_max_path()

    enum: TILEDB_OFFSET_SIZE
    unsigned int tiledb_offset_size()

    # Version
    void tiledb_version(int* major, int* minor, int* rev)

    # Stats
    void tiledb_stats_enable()
    void tiledb_stats_disable()
    void tiledb_stats_reset()
    int32_t tiledb_stats_dump_str(char** out)
    int32_t tiledb_stats_raw_dump_str(char** out)
    int32_t tiledb_stats_free_str(char** out)

    # Enums
    ctypedef enum tiledb_object_t:
        TILEDB_INVALID
        TILEDB_GROUP
        TILEDB_ARRAY

    ctypedef enum tiledb_query_type_t:
        TILEDB_READ
        TILEDB_WRITE

    ctypedef enum tiledb_query_status_t:
        TILEDB_FAILED
        TILEDB_COMPLETED
        TILEDB_INPROGRESS
        TILEDB_INCOMPLETE

    ctypedef enum tiledb_datatype_t:
        TILEDB_INT32
        TILEDB_INT64
        TILEDB_FLOAT32
        TILEDB_FLOAT64
        TILEDB_CHAR
        TILEDB_INT8
        TILEDB_UINT8
        TILEDB_INT16
        TILEDB_UINT16
        TILEDB_UINT32
        TILEDB_UINT64
        TILEDB_STRING_ASCII
        TILEDB_STRING_UTF8
        TILEDB_STRING_UTF16
        TILEDB_STRING_UTF32
        TILEDB_STRING_UCS2
        TILEDB_STRING_UCS4
        TILEDB_DATETIME_YEAR
        TILEDB_DATETIME_MONTH
        TILEDB_DATETIME_WEEK
        TILEDB_DATETIME_DAY
        TILEDB_DATETIME_HR
        TILEDB_DATETIME_MIN
        TILEDB_DATETIME_SEC
        TILEDB_DATETIME_MS
        TILEDB_DATETIME_US
        TILEDB_DATETIME_NS
        TILEDB_DATETIME_PS
        TILEDB_DATETIME_FS
        TILEDB_DATETIME_AS

    ctypedef enum tiledb_array_type_t:
        TILEDB_DENSE
        TILEDB_SPARSE

    ctypedef enum tiledb_layout_t:
        TILEDB_ROW_MAJOR
        TILEDB_COL_MAJOR
        TILEDB_GLOBAL_ORDER
        TILEDB_UNORDERED
        TILEDB_HILBERT

    ctypedef enum tiledb_filter_type_t:
        TILEDB_FILTER_NONE = 0
        TILEDB_FILTER_GZIP = 1
        TILEDB_FILTER_ZSTD = 2
        TILEDB_FILTER_LZ4 = 3
        TILEDB_FILTER_RLE = 4
        TILEDB_FILTER_BZIP2 = 5
        TILEDB_FILTER_DOUBLE_DELTA = 6
        TILEDB_FILTER_BIT_WIDTH_REDUCTION = 7
        TILEDB_FILTER_BITSHUFFLE = 8
        TILEDB_FILTER_BYTESHUFFLE = 9
        TILEDB_FILTER_POSITIVE_DELTA = 10
        # 11 is encryption, see tiledb_enum.h
        TILEDB_FILTER_CHECKSUM_MD5 = 12
        TILEDB_FILTER_CHECKSUM_SHA256 = 13

    ctypedef enum tiledb_filter_option_t:
        TILEDB_COMPRESSION_LEVEL = 0
        TILEDB_BIT_WIDTH_MAX_WINDOW = 1
        TILEDB_POSITIVE_DELTA_MAX_WINDOW = 2

    ctypedef enum tiledb_encryption_type_t:
        TILEDB_NO_ENCRYPTION
        TILEDB_AES_256_GCM

    ctypedef enum tiledb_walk_order_t:
        TILEDB_PREORDER
        TILEDB_POSTORDER

    ctypedef enum tiledb_filesystem_t:
        TILEDB_HDFS
        TILEDB_S3
        TILEDB_AZURE
        TILEDB_GCS

    ctypedef enum tiledb_vfs_mode_t:
        TILEDB_VFS_READ
        TILEDB_VFS_WRITE
        TILEDB_VFS_APPEND

    # Types
    ctypedef struct tiledb_ctx_t:
        pass
    ctypedef struct tiledb_config_t:
        pass
    ctypedef struct tiledb_config_iter_t:
        pass
    ctypedef struct tiledb_error_t:
        pass
    ctypedef struct tiledb_array_t:
        pass
    ctypedef struct tiledb_attribute_t:
        pass
    ctypedef struct tiledb_array_schema_t:
        pass
    ctypedef struct tiledb_dimension_t:
        pass
    ctypedef struct tiledb_domain_t:
        pass
    ctypedef struct tiledb_query_t:
        pass
    ctypedef struct tiledb_filter_t:
        pass
    ctypedef struct tiledb_filter_list_t:
        pass
    ctypedef struct tiledb_vfs_t:
        pass
    ctypedef struct tiledb_vfs_fh_t:
        pass

    # Config
    int tiledb_config_alloc(
        tiledb_config_t** config,
        tiledb_error_t** error)

    void tiledb_config_free(
        tiledb_config_t** config)

    int tiledb_config_set(
        tiledb_config_t* config,
        const char* param,
        const char* value,
        tiledb_error_t** error)

    int tiledb_config_get(
        tiledb_config_t* config,
        const char* param,
        const char** value,
        tiledb_error_t** error)

    int tiledb_config_load_from_file(
        tiledb_config_t* config,
        const char* filename,
        tiledb_error_t** error) nogil

    int tiledb_config_unset(
        tiledb_config_t* config,
        const char* param,
        tiledb_error_t** error)

    int tiledb_config_save_to_file(
        tiledb_config_t* config,
        const char* filename,
        tiledb_error_t** error) nogil

    # Config Iterator
    int tiledb_config_iter_alloc(
        tiledb_config_t* config,
        const char* prefix,
        tiledb_config_iter_t** config_iter,
        tiledb_error_t** error)

    void tiledb_config_iter_free(
        tiledb_config_iter_t** config_iter)

    int tiledb_config_iter_here(
        tiledb_config_iter_t* config_iter,
        const char** param,
        const char** value,
        tiledb_error_t** error)

    int tiledb_config_iter_next(
        tiledb_config_iter_t* config_iter,
        tiledb_error_t** error)

    int tiledb_config_iter_done(
        tiledb_config_iter_t* config_iter,
        int* done,
        tiledb_error_t** error)

    # Context
    int tiledb_ctx_alloc(
        tiledb_config_t* config,
        tiledb_ctx_t** ctx)

    void tiledb_ctx_free(
        tiledb_ctx_t** ctx)

    int tiledb_ctx_get_config(
        tiledb_ctx_t* ctx,
        tiledb_config_t** config)

    int tiledb_ctx_get_last_error(
        tiledb_ctx_t* ctx,
        tiledb_error_t** error)

    int tiledb_ctx_get_stats(
        tiledb_ctx_t* ctx,
        char** stats_json);

    int tiledb_ctx_is_supported_fs(
        tiledb_ctx_t* ctx,
        tiledb_filesystem_t fs,
        int* is_supported)

    int tiledb_ctx_set_tag(
        tiledb_ctx_t* ctx,
        const char* key,
        const char* value)

    # Error
    int tiledb_error_message(
        tiledb_error_t* err,
        char** msg)

    void tiledb_error_free(
        tiledb_error_t** err)

    # Group
    int tiledb_group_create(
        tiledb_ctx_t* ctx,
        const char* group) nogil

    # Filter
    int tiledb_filter_alloc(
        tiledb_ctx_t* ctx,
        tiledb_filter_type_t filter_type,
        tiledb_filter_t** filter) nogil

    int tiledb_filter_free(
        tiledb_filter_t **filter)

    int tiledb_filter_get_type(
        tiledb_ctx_t* ctx,
        tiledb_filter_t* filter,
        tiledb_filter_type_t* type)

    int tiledb_filter_set_option(
        tiledb_ctx_t* ctx,
        tiledb_filter_t* filter,
        tiledb_filter_option_t option,
        const void* value)

    int tiledb_filter_get_option(
        tiledb_ctx_t* ctx,
        tiledb_filter_t* filter,
        tiledb_filter_option_t option,
        void* value)

    # Filter List
    int tiledb_filter_list_alloc(
        tiledb_ctx_t* ctx,
        tiledb_filter_list_t** filter_list)

    int tiledb_filter_list_free(
        tiledb_filter_list_t** filter_list)

    int tiledb_filter_list_add_filter(
        tiledb_ctx_t* ctx,
        tiledb_filter_list_t* filter_list,
        tiledb_filter_t* filter)

    int tiledb_filter_list_set_max_chunk_size(
        tiledb_ctx_t* ctx,
        const tiledb_filter_list_t* filter_list,
        unsigned int max_chunk_size)

    int tiledb_filter_list_get_nfilters(
        tiledb_ctx_t* ctx,
        const tiledb_filter_list_t* filter_list,
        unsigned int* num_filters)

    int tiledb_filter_list_get_filter_from_index(
        tiledb_ctx_t* ctx,
        const tiledb_filter_list_t* filter_list,
        unsigned int index,
        tiledb_filter_t** filter)

    int tiledb_filter_list_get_max_chunk_size(
        tiledb_ctx_t* ctx,
        const tiledb_filter_list_t* filter_list,
        unsigned int* max_chunk_size)

    # Attribute
    int tiledb_attribute_alloc(
        tiledb_ctx_t* ctx,
        const char* name,
        tiledb_datatype_t atype,
        tiledb_attribute_t** attr)

    void tiledb_attribute_free(
        tiledb_attribute_t** attr)

    int tiledb_attribute_set_filter_list(
        tiledb_ctx_t* ctx_ptr,
        const tiledb_attribute_t* attr,
        tiledb_filter_list_t* filter_list)

    int tiledb_attribute_set_fill_value(
        tiledb_ctx_t *ctx,
        tiledb_attribute_t *attr,
        const void *value, uint64_t size)

    int tiledb_attribute_set_cell_val_num(
        tiledb_ctx_t* ctx,
        tiledb_attribute_t* attr,
        unsigned int cell_val_num)

    int tiledb_attribute_get_name(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        const char** name)

    int tiledb_attribute_get_type(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        tiledb_datatype_t* type)

    int tiledb_attribute_get_filter_list(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        tiledb_filter_list_t** filter_list)

    int tiledb_attribute_get_fill_value(
        tiledb_ctx_t *ctx,
        tiledb_attribute_t *attr,
        const void **value,
        uint64_t *size)

    int tiledb_attribute_get_cell_val_num(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        unsigned int* cell_val_num)

    int tiledb_attribute_set_nullable(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        uint8_t nullable)

    int tiledb_attribute_get_nullable(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        uint8_t* nullable)

    int tiledb_attribute_dump(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        FILE* out)

    # Datatype
    uint64_t tiledb_datatype_size(
        tiledb_datatype_t type);

    # Domain
    int tiledb_domain_alloc(
        tiledb_ctx_t* ctx,
        tiledb_domain_t** domain)

    void tiledb_domain_free(
        tiledb_domain_t** domain)

    int tiledb_domain_get_type(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        tiledb_datatype_t* dtype)

    int tiledb_domain_get_ndim(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        unsigned int* ndim)

    int tiledb_domain_add_dimension(
        tiledb_ctx_t* ctx,
        tiledb_domain_t* domain,
        tiledb_dimension_t* dim)

    int tiledb_domain_get_dimension_from_index(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        unsigned int index,
        tiledb_dimension_t** dim)

    int tiledb_domain_get_dimension_from_name(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        const char* name,
        tiledb_dimension_t** dim)

    int tiledb_domain_has_dimension(
        tiledb_ctx_t * ctx,
        const tiledb_domain_t* domain,
        const char* name,
        int32_t* has_dim)

    int tiledb_domain_dump(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        FILE* out)

    # Dimension
    int tiledb_dimension_alloc(
        tiledb_ctx_t* ctx,
        const char* name,
        tiledb_datatype_t type,
        const void* dim_domain,
        const void* tile_extent,
        tiledb_dimension_t** dim)

    void tiledb_dimension_free(
        tiledb_dimension_t** dim)

    int tiledb_dimension_get_name(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        const char** name)

    int tiledb_dimension_get_cell_val_num(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        uint32_t* cell_val_num)

    int tiledb_dimension_get_type(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        tiledb_datatype_t* type)

    int tiledb_dimension_get_domain(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        const void** domain)

    int tiledb_dimension_get_tile_extent(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        const void** tile_extent)

    int tiledb_dimension_get_filter_list(
        tiledb_ctx_t *ctx,
        tiledb_dimension_t *dim,
        tiledb_filter_list_t **filter_list)

    int tiledb_dimension_set_filter_list(
        tiledb_ctx_t *ctx,
        tiledb_dimension_t *dim,
        tiledb_filter_list_t *filter_list)

    # Array schema
    int tiledb_array_schema_alloc(
        tiledb_ctx_t* ctx,
        tiledb_array_type_t array_type,
        tiledb_array_schema_t** array_schema)

    void tiledb_array_schema_free(
        tiledb_array_schema_t** array_schema)

    int tiledb_array_schema_add_attribute(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_attribute_t* attr)

    int tiledb_array_schema_has_attribute(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        const char* name,
        int32_t* has_attr)

    int tiledb_array_schema_set_domain(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_domain_t* domain);

    int tiledb_array_schema_set_capacity(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        uint64_t capacity);

    int tiledb_array_schema_set_cell_order(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_layout_t cell_order);

    int tiledb_array_schema_set_tile_order(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_layout_t tile_order)

    int tiledb_array_schema_set_offsets_filter_list(
        tiledb_ctx_t* ctx,
         tiledb_array_schema_t* array_schmea,
        tiledb_filter_list_t* filter_list)

    int tiledb_array_schema_set_coords_filter_list(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_filter_list_t* filter_list)

    int tiledb_array_schema_check(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema)

    int tiledb_array_schema_load(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        tiledb_array_schema_t** array_schema) nogil

    int tiledb_array_schema_load_with_key(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        tiledb_encryption_type_t key_type,
        const void* key_ptr,
        unsigned int key_len,
        tiledb_array_schema_t** array_schema) nogil

    int tiledb_array_schema_get_array_type(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_array_type_t* array_type)

    int tiledb_array_schema_get_capacity(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        uint64_t* capacity)

    int tiledb_array_schema_get_cell_order(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_layout_t* cell_order)

    int tiledb_array_schema_get_coords_filter_list(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_filter_list_t** filter_list)

    int tiledb_array_schema_get_offsets_filter_list(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_filter_list_t** filter_list)

    int tiledb_array_schema_get_domain(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_domain_t** domain)

    int tiledb_array_schema_get_tile_order(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_layout_t* tile_order)

    int tiledb_array_schema_get_attribute_num(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        unsigned int* num_attributes)

    int tiledb_array_schema_get_attribute_from_index(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        unsigned int index,
        tiledb_attribute_t** attr)

    int tiledb_array_schema_get_attribute_from_name(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        const char* name,
        tiledb_attribute_t** attr)

    int tiledb_array_schema_get_array_name(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        const char** array_name)

    int tiledb_array_schema_get_allows_dups(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        int* allows_dups);

    int tiledb_array_schema_set_allows_dups(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        int allows_dups)

    int tiledb_array_schema_dump(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        FILE* out)

    int tiledb_array_get_open_timestamp_start(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t* timestamp_start)

    int tiledb_array_get_open_timestamp_end(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t* timestamp_end)

    int tiledb_array_set_open_timestamp_start(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t timestamp_start)

    int tiledb_array_set_open_timestamp_end(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t timestamp_end)

    int tiledb_array_put_metadata(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* key,
        tiledb_datatype_t value_type,
        uint32_t value_num,
        const void* value)

    int tiledb_array_delete_metadata(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* key)

    int tiledb_array_has_metadata_key(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* key,
        tiledb_datatype_t* value_type,
        int32_t* has_key)

    int tiledb_array_get_metadata(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* key,
        tiledb_datatype_t* value_type,
        uint32_t* value_num,
        const void** value)

    int tiledb_array_get_metadata_num(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t* num)

    int tiledb_array_get_metadata_from_index(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t index,
        const char** key,
        uint32_t* key_len,
        tiledb_datatype_t* value_type,
        uint32_t* value_num,
        const void** value)

    int tiledb_array_consolidate_metadata(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        tiledb_config_t* config)

    int tiledb_array_consolidate_metadata_with_key(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        tiledb_encryption_type_t encryption_type,
        const void* encryption_key,
        uint32_t key_length,
        tiledb_config_t* config)

    # Query
    int tiledb_query_alloc(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        tiledb_query_type_t query_type,
        tiledb_query_t** query)

    int tiledb_query_set_subarray(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const void* subarray)

    int tiledb_query_set_buffer(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* attribute,
        void* buffer,
        uint64_t* buffer_size)

    int32_t tiledb_query_set_data_buffer(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* name,
        void* buffer,
        uint64_t* buffer_size)

    int tiledb_query_set_validity_buffer(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* name,
        uint8_t* buffer,
        uint64_t* buffer_size)

    int32_t tiledb_query_set_offsets_buffer(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* name,
        uint64_t* buffer,
        uint64_t* buffer_size)

    int tiledb_query_set_buffer_var(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* attribute,
        uint64_t* buffer_off,
        uint64_t* buffer_off_size,
        void* buffer_val,
        uint64_t* buffer_val_size)

    int tiledb_query_set_buffer_nullable(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* name,
        void* buffer,
        uint64_t* buffer_size,
        uint8_t* buffer_validity_bytemap,
        uint64_t* buffer_validity_bytemap_size)

    int tiledb_query_set_buffer_var_nullable(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char* name,
        uint64_t* buffer_off,
        uint64_t* buffer_off_size,
        void* buffer_val,
        uint64_t* buffer_val_size,
        uint8_t* buffer_validity_bytemap,
        uint64_t* buffer_validity_bytemap_size)

    int tiledb_query_set_layout(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        tiledb_layout_t layout)

    void tiledb_query_free(
        tiledb_query_t** query)

    int tiledb_query_finalize(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query) nogil

    int tiledb_query_submit(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query) nogil

    int tiledb_query_submit_async(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        void* (*callback)(void*),
        void* callback_data)

    int tiledb_query_get_status(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        tiledb_query_status_t* status)

    int tiledb_query_get_type(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        tiledb_query_type_t* query_type)

    int tiledb_query_has_results(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        int* has_results)

    int tiledb_query_get_fragment_num(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        uint32_t* num)

    int tiledb_query_get_fragment_uri(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        uint64_t idx,
        const char** uri)

    int tiledb_query_get_fragment_timestamp_range(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        uint64_t idx,
        uint64_t* t1,
        uint64_t* t2)

    int tiledb_query_add_range(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        uint32_t dim_idx,
        const void * start,
        const void * end,
        const void * stride)

    int tiledb_query_add_range_var(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        uint32_t dim_idx,
        const void * start,
        uint64_t start_size,
        const void * end,
        uint64_t end_size)

    int tiledb_query_get_range(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        uint32_t dim_idx,
        uint64_t range_idx,
        const void** start,
        const void** end,
        const void** stride)

    int tiledb_query_get_range_num(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        uint32_t dim_idx,
        uint64_t * range_num)

    int tiledb_query_get_est_result_size(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        const char* attr_name,
        uint64_t* size)

    int tiledb_query_get_est_result_size_var(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        const char* attr_name,
        uint64_t* size_off,
        uint64_t* size_val)

    int tiledb_query_get_stats(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        char** stats_json);

    # Array
    int tiledb_array_alloc(
        tiledb_ctx_t* ctx,
        const char* uri,
        tiledb_array_t** array)

    int tiledb_array_open(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        tiledb_query_type_t query_type) nogil

    int tiledb_array_open_with_key(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        tiledb_query_type_t query_type,
        tiledb_encryption_type_t key_type,
        const void* key,
        unsigned int key_len) nogil

    int tiledb_array_open_at_with_key(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        tiledb_query_type_t query_type,
        tiledb_encryption_type_t encryption_type,
        const void * encryption_key,
        int key_length,
        uint64_t timestamp) nogil

    int tiledb_array_reopen(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array) nogil

    int tiledb_array_reopen_at(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t timestamp) nogil

    int tiledb_array_close(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array) nogil

    void tiledb_array_free(
        tiledb_array_t** array)

    int tiledb_array_create(
        tiledb_ctx_t* ctx,
        const char* uri,
        const tiledb_array_schema_t* array_schema) nogil

    int tiledb_array_create_with_key(
        tiledb_ctx_t* ctx,
        const char* uri,
        const tiledb_array_schema_t* array_schema,
        tiledb_encryption_type_t key_type,
        const void* key,
        unsigned int key_len) nogil

    int tiledb_array_is_open(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        int* is_open)

    int tiledb_array_consolidate(
        tiledb_ctx_t* ctx,
        const char* array_path,
        tiledb_config_t* config) nogil

    int tiledb_array_consolidate_with_key(
        tiledb_ctx_t* ctx,
        const char* uri,
        tiledb_encryption_type_t key_type,
        const void* key_ptr,
        unsigned int key_len,
        tiledb_config_t* config) nogil

    int tiledb_array_get_schema(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        tiledb_array_schema_t** array_schema) nogil

    int tiledb_array_get_timestamp(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint64_t* timestamp) nogil

    int tiledb_array_get_query_type(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        tiledb_query_type_t* query_type)

    int tiledb_array_get_non_empty_domain(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        void* domain,
        int* isempty) nogil

    int32_t tiledb_array_get_non_empty_domain_from_index(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint32_t idx,
        void* domain,
        int32_t* is_empty);

    int32_t tiledb_array_get_non_empty_domain_from_name(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* name,
        void* domain,
        int32_t* is_empty);

    int32_t tiledb_array_get_non_empty_domain_var_size_from_index(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint32_t idx,
        uint64_t* start_size,
        uint64_t* end_size,
        int32_t* is_empty)

    int32_t tiledb_array_get_non_empty_domain_var_size_from_name(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* name,
        uint64_t* start_size,
        uint64_t* end_size,
        int32_t* is_empty)

    int32_t tiledb_array_get_non_empty_domain_var_from_index(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        uint32_t idx,
        void* start,
        void* end,
        int32_t* is_empty);

    int32_t tiledb_array_get_non_empty_domain_var_from_name(
        tiledb_ctx_t* ctx,
        tiledb_array_t* array,
        const char* name,
        void* start,
        void* end,
        int32_t* is_empty)

    int tiledb_array_vacuum(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        tiledb_config_t* config) nogil

    # Resource management
    int tiledb_object_type(
        tiledb_ctx_t* ctx,
        const char* path,
        tiledb_object_t* otype) nogil

    int tiledb_object_remove(
        tiledb_ctx_t* ctx,
        const char* path) nogil

    int tiledb_object_move(
        tiledb_ctx_t* ctx,
        const char* old_path,
        const char* new_path) nogil

    int tiledb_object_walk(
        tiledb_ctx_t* ctx,
        const char* path,
        tiledb_walk_order_t order,
        int (*callback)(const char*, tiledb_object_t, void*),
        void* data)

    int tiledb_object_ls(
        tiledb_ctx_t* ctx,
        const char* path,
        int (*callback)(const char*, tiledb_object_t, void*),
        void* data)

    # VFS
    int tiledb_vfs_alloc(
        tiledb_ctx_t* ctx,
        tiledb_config_t* config,
        tiledb_vfs_t** vfs)

    void tiledb_vfs_free(
        tiledb_vfs_t** vfs)

    int tiledb_vfs_create_bucket(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_remove_bucket(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_empty_bucket(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_is_empty_bucket(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri,
        int* is_empty) nogil

    int tiledb_vfs_is_bucket(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri,
        int* is_bucket) nogil

    int tiledb_vfs_create_dir(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_is_dir(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri,
        int* is_dir) nogil

    int tiledb_vfs_remove_dir(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_is_file(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri,
        int* is_file) nogil

    int tiledb_vfs_remove_file(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_file_size(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri,
        uint64_t* size) nogil

    int tiledb_vfs_dir_size(
        tiledb_ctx_t * ctx,
        tiledb_vfs_t * vfs,
        const char * uri,
        uint64_t * size) nogil

    int tiledb_vfs_ls(
        tiledb_ctx_t * ctx,
        tiledb_vfs_t * vfs,
        const char * path,
        int (*callback)(const char *, void *),
        void * data)

    int tiledb_vfs_move_file(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* old_uri,
        const char* new_uri) nogil

    int tiledb_vfs_move_dir(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* old_uri,
        const char* new_uri) nogil

    int tiledb_vfs_copy_file(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* old_uri,
        const char* new_uri) nogil

    int tiledb_vfs_copy_dir(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* old_uri,
        const char* new_uri) nogil

    int tiledb_vfs_open(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri,
        tiledb_vfs_mode_t mode,
        tiledb_vfs_fh_t** fh) nogil

    int tiledb_vfs_close(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t* fh) nogil

    int tiledb_vfs_read(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t* fh,
        uint64_t offset,
        void* buffer,
        uint64_t nbytes) nogil

    int tiledb_vfs_write(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t* fh,
        const void* buffer,
        uint64_t nbytes) nogil

    int tiledb_vfs_sync(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t* fh) nogil

    void tiledb_vfs_fh_free(
        tiledb_vfs_fh_t** fh) nogil

    int tiledb_vfs_fh_is_closed(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t* fh,
        int* is_closed) nogil

    int tiledb_vfs_touch(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    int tiledb_vfs_get_config(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        tiledb_config_t** config)

    # URI
    int tiledb_uri_to_path(
        tiledb_ctx_t* ctx,
        const char* uri,
        char* path_out,
        unsigned* path_length) nogil

# Free helper functions
cpdef unicode ustring(object s)
cpdef check_error(Ctx ctx, int rc)
cdef _raise_tiledb_error(tiledb_error_t* err_ptr)
cdef _raise_ctx_err(tiledb_ctx_t* ctx_ptr, int rc)

cdef tiledb_datatype_t _tiledb_dtype_datetime(np.dtype dtype) except? TILEDB_DATETIME_YEAR

###############################################################################
#                                                                             #
#   TileDB-Py API declaration                                                 #
#                                                                             #
###############################################################################

cdef class Config(object):
    cdef tiledb_config_t* ptr

    @staticmethod
    cdef from_ptr(tiledb_config_t* ptr)

cdef class ConfigKeys(object):
    cdef ConfigItems config_items

cdef class ConfigItems(object):
    cdef Config config
    cdef tiledb_config_iter_t* ptr

cdef class ConfigValues(object):
    cdef ConfigItems config_items

cdef class Ctx(object):
    cdef tiledb_ctx_t* ptr

cdef class Filter(object):
    cdef Ctx ctx
    cdef tiledb_filter_t* ptr

cdef class FilterList(object):
    cdef Ctx ctx
    cdef tiledb_filter_list_t* ptr

    @staticmethod
    cdef FilterList from_ptr(tiledb_filter_list_t* ptr, Ctx ctx=*)

    cdef Filter _getfilter(FilterList self, int idx)

cdef class Attr(object):
    cdef Ctx ctx
    cdef tiledb_attribute_t* ptr

    @staticmethod
    cdef from_ptr(const tiledb_attribute_t* ptr, Ctx ctx=*)
    cdef unicode _get_name(Attr self)
    cdef unsigned int _cell_val_num(Attr self) except? 0
    cdef tiledb_datatype_t _get_type(Attr self) except? TILEDB_CHAR


cdef class Dim(object):
    cdef Ctx ctx
    cdef tiledb_dimension_t* ptr

    @staticmethod
    cdef from_ptr(const tiledb_dimension_t* ptr, Ctx ctx=*)

    cdef tiledb_datatype_t _get_type(Dim self) except? TILEDB_CHAR
    cdef unsigned int _cell_val_num(Dim self) except? 0
    cdef _integer_domain(self)
    cdef _datetime_domain(self)
    cdef _shape(self)

cdef class Domain(object):
    cdef Ctx ctx
    cdef tiledb_domain_t* ptr

    @staticmethod
    cdef from_ptr(const tiledb_domain_t* ptr, Ctx ctx=*)
    cdef tiledb_datatype_t _get_type(Domain self) except? TILEDB_CHAR
    cdef _integer_domain(Domain self)
    cdef _is_homogeneous(Domain self)
    cdef _shape(Domain self)

cdef class ArraySchema(object):
    cdef Ctx ctx
    cdef tiledb_array_schema_t* ptr

    @staticmethod
    cdef from_ptr(const tiledb_array_schema_t* schema_ptr, Ctx ctx=*)
    cdef _cell_order(ArraySchema self, tiledb_layout_t* cell_order_ptr)
    cdef _tile_order(ArraySchema self, tiledb_layout_t* tile_order_ptr)
    cdef _attr_name(self, name)
    cdef _attr_idx(self, int idx)

cdef class Array(object):
    cdef object __weakref__
    cdef Ctx ctx
    cdef tiledb_array_t* ptr
    cdef unicode uri
    cdef unicode mode
    cdef bint _isopen
    cdef object view_attr # can be None
    cdef object key # can be None
    cdef object schema
    cdef object _buffers

    cdef DomainIndexer domain_index
    cdef object multi_index
    cdef object df
    cdef Metadata meta
    cdef object last_fragment_info
    cdef object pyquery

    cdef _ndarray_is_varlen(self, np.ndarray array)

cdef class SparseArrayImpl(Array):
    cdef _read_sparse_subarray(self, list subarray, list attr_names, object attr_cond, tiledb_layout_t layout)

cdef class DenseArrayImpl(Array):
    cdef _read_dense_subarray(self, list subarray, list attr_names, object attr_cond, tiledb_layout_t layout, bint include_coords)

cdef class FileHandle(object):
    cdef Ctx ctx
    cdef VFS vfs
    cdef unicode uri
    cdef tiledb_vfs_fh_t* ptr

    @staticmethod
    cdef from_ptr(VFS vfs, unicode uri, tiledb_vfs_fh_t* fh_ptr)
    cpdef closed(self)

cdef class VFS(object):
    cdef Ctx ctx
    cdef tiledb_vfs_t* ptr

cdef class Query(object):
    cdef Array array
    cdef object attrs
    cdef object attr_cond
    cdef object dims
    cdef object order
    cdef object coords
    cdef object index_col
    cdef object use_arrow
    cdef object return_arrow
    cdef object return_incomplete
    cdef DomainIndexer domain_index
    cdef object multi_index
    cdef object df

cdef class ReadQuery(object):
    cdef object _buffers
    cdef object _offsets

cdef class Metadata(object):
    cdef object array_ref

cdef class TileDBError(Exception):
    pass


IF (not TILEDBPY_MODULAR):
    include "indexing.pxd"
