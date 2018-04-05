from libc.stdio cimport FILE
from libc.stdint cimport uint64_t

cdef extern from "tiledb/tiledb.h":
    # Constants
    enum: TILEDB_OK
    enum: TILEDB_ERR
    enum: TILEDB_OOM

    enum: TILEDB_VAR_NUM
    unsigned int tiledb_var_num()

    # TILEDB_COORDS
    const char* tiledb_coords()

    enum: TILEDB_MAX_PATH
    unsigned int tiledb_max_path()

    # Version
    void tiledb_version(int* major, int* minor, int* rev)

    # Stats
    void tiledb_stats_enable()
    void tiledb_stats_disable()
    void tiledb_stats_reset()
    void tiledb_stats_dump(FILE* out)

    # Enums
    ctypedef enum tiledb_object_t:
        TILEDB_INVALID
        TILEDB_GROUP
        TILEDB_ARRAY
        TILEDB_KEY_VALUE

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

    ctypedef enum tiledb_array_type_t:
        TILEDB_DENSE
        TILEDB_SPARSE

    ctypedef enum tiledb_layout_t:
        TILEDB_ROW_MAJOR
        TILEDB_COL_MAJOR
        TILEDB_GLOBAL_ORDER
        TILEDB_UNORDERED

    ctypedef enum tiledb_compressor_t:
        TILEDB_NO_COMPRESSION
        TILEDB_GZIP
        TILEDB_ZSTD
        TILEDB_LZ4
        TILEDB_BLOSC_LZ
        TILEDB_BLOSC_LZ4
        TILEDB_BLOSC_LZ4HC
        TILEDB_BLOSC_SNAPPY
        TILEDB_BLOSC_ZSTD
        TILEDB_RLE
        TILEDB_BZIP2
        TILEDB_DOUBLE_DELTA

    ctypedef enum tiledb_walk_order_t:
        TILEDB_PREORDER
        TILEDB_POSTORDER

    ctypedef enum tiledb_filesystem_t:
        TILEDB_HDFS
        TILEDB_S3

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
    ctypedef struct tiledb_kv_schema_t:
        pass
    ctypedef struct tiledb_kv_t:
        pass
    ctypedef struct tiledb_kv_item_t:
        pass
    ctypedef struct tiledb_kv_iter_t:
        pass
    ctypedef struct tiledb_vfs_t:
        pass
    ctypedef struct tiledb_vfs_fh_t:
        pass

    # Config
    int tiledb_config_create(
        tiledb_config_t** config,
        tiledb_error_t** error)

    int tiledb_config_free(
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
    int tiledb_config_iter_create(
        tiledb_config_t* config,
        tiledb_config_iter_t** config_iter,
        const char* prefix,
        tiledb_error_t** error)

    int tiledb_config_iter_free(
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
    int tiledb_ctx_create(
        tiledb_ctx_t** ctx,
        tiledb_config_t* config)

    int tiledb_ctx_free(
        tiledb_ctx_t** ctx)

    int tiledb_ctx_get_config(
        tiledb_ctx_t* ctx,
        tiledb_config_t** config)

    int tiledb_ctx_get_last_error(
        tiledb_ctx_t* ctx,
        tiledb_error_t** error)

    int tiledb_ctx_is_supported_fs(
        tiledb_ctx_t* ctx,
        tiledb_filesystem_t fs,
        int* is_supported)

    # Error
    int tiledb_error_message(
        tiledb_error_t* err,
        char** msg)

    int tiledb_error_free(
        tiledb_error_t** err)

    # Group
    int tiledb_group_create(
        tiledb_ctx_t* ctx,
        const char* group) nogil

    # Attribute
    int tiledb_attribute_create(
        tiledb_ctx_t* ctx,
        tiledb_attribute_t** attr,
        const char* name,
        tiledb_datatype_t atype)

    int tiledb_attribute_free(
        tiledb_ctx_t* ctx,
        tiledb_attribute_t** attr)

    int tiledb_attribute_set_compressor(
        tiledb_ctx_t* ctx,
        tiledb_attribute_t* attr,
        tiledb_compressor_t compressor,
        int compression_level)

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
        tiledb_datatype_t* type);

    int tiledb_attribute_get_compressor(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        tiledb_compressor_t* compressor,
        int* compression_level)

    int tiledb_attribute_get_cell_val_num(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        unsigned int* cell_val_num)


    int tiledb_attribute_dump(
        tiledb_ctx_t* ctx,
        const tiledb_attribute_t* attr,
        FILE* out)

    # Domain
    int tiledb_domain_create(
        tiledb_ctx_t* ctx,
        tiledb_domain_t** domain)

    int tiledb_domain_free(
        tiledb_ctx_t* ctx,
        tiledb_domain_t** domain)

    int tiledb_domain_get_type(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        tiledb_datatype_t* dtype)

    int tiledb_domain_get_rank(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        unsigned int* rank)

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

    int tiledb_domain_dump(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        FILE* out)

    # Dimension
    int tiledb_dimension_create(
        tiledb_ctx_t* ctx,
        tiledb_dimension_t** dim,
        const char* name,
        tiledb_datatype_t type,
        const void* dim_domain,
        const void* tile_extent)

    int tiledb_dimension_free(
        tiledb_ctx_t* ctx,
        tiledb_dimension_t** dim)

    int tiledb_dimension_get_name(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        const char** name)

    int tiledb_dimension_get_type(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        tiledb_datatype_t* type)

    int tiledb_dimension_get_domain(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        void** domain)

    int tiledb_dimension_get_tile_extent(
        tiledb_ctx_t* ctx,
        const tiledb_dimension_t* dim,
        void** tile_extent)

    # Array schema
    int tiledb_array_schema_create(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t** array_schema,
        tiledb_array_type_t array_type)

    int tiledb_array_schema_free(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t** array_schema)

    int tiledb_array_schema_add_attribute(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_attribute_t* attr)

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

    int tiledb_array_schema_set_coords_compressor(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_compressor_t compressor,
        int compression_level)

    int tiledb_array_schema_set_offsets_compressor(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_compressor_t compressor,
        int compression_level)

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

    int tiledb_array_schema_check(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema)

    int tiledb_array_schema_load(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t** array_schema,
        const char* array_name) nogil

    int tiledb_array_schema_get_array_name(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        const char** array_name)

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
        tiledb_layout_t* cell_order);

    int tiledb_array_schema_get_coords_compressor(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_compressor_t* compressor,
        int* compression_level)

    int tiledb_array_schema_get_offsets_compressor(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_compressor_t* compressor,
        int* compression_level)

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

    int tiledb_array_schema_dump(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        FILE* out)

    # Query
    int tiledb_query_create(
        tiledb_ctx_t* ctx,
        tiledb_query_t** query,
        const char* array_name,
        tiledb_query_type_t qtype)

    int tiledb_query_set_subarray(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const void* subarray)

    int tiledb_query_set_buffers(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const char** attributes,
        unsigned int attribute_num,
        void** buffers,
        uint64_t* buffer_sizes)

    int tiledb_query_reset_buffers(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        void** buffers,
        uint64_t* buffer_sizes)

    int tiledb_query_set_layout(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        tiledb_layout_t layout)

    int tiledb_query_free(
        tiledb_ctx_t* ctx,
        tiledb_query_t** query)

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

    int tiledb_query_get_attribute_status(
        tiledb_ctx_t* ctx,
        const tiledb_query_t* query,
        const char* attribute_name,
        tiledb_query_status_t* status)

    # Array
    int tiledb_array_create(
        tiledb_ctx_t* ctx,
        const char* uri,
        const tiledb_array_schema_t* array_schema) nogil

    int tiledb_array_consolidate(
        tiledb_ctx_t* ctx,
        const char* array_path) nogil

    int tiledb_array_get_non_empty_domain(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        void* domain,
        int* isempty)

    int tiledb_array_compute_max_read_buffer_sizes(
        tiledb_ctx_t* ctx,
        const char* array_uri,
        const void* subarray,
        const char** attributes,
        unsigned attribute_num,
        uint64_t* buffer_sizes);

    # Key / Value Schema
    int tiledb_kv_schema_create(
        tiledb_ctx_t* ctx,
        tiledb_kv_schema_t** kv_schema)

    int tiledb_kv_schema_free(
        tiledb_ctx_t* ctx,
        tiledb_kv_schema_t** kv_schema)

    int tiledb_kv_schema_add_attribute(
        tiledb_ctx_t* ctx,
        tiledb_kv_schema_t* kv_schema,
        tiledb_attribute_t* attr)

    int tiledb_kv_schema_check(
        tiledb_ctx_t* ctx,
        tiledb_kv_schema_t* kv_schema)

    int tiledb_kv_schema_load(
        tiledb_ctx_t* ctx,
        tiledb_kv_schema_t** kv_schema,
        const char* uri) nogil

    int tiledb_kv_schema_get_attribute_num(
        tiledb_ctx_t* ctx,
        const tiledb_kv_schema_t* kv_schema,
        unsigned int* attribute_num)

    int tiledb_kv_schema_get_attribute_from_index(
        tiledb_ctx_t* ctx,
        const tiledb_kv_schema_t* kv_schema,
        unsigned int index,
        tiledb_attribute_t** attr)

    int tiledb_kv_schema_get_attribute_from_name(
        tiledb_ctx_t* ctx,
        const tiledb_kv_schema_t* kv_schema,
        const char* name,
        tiledb_attribute_t** attr)

    int tiledb_kv_schema_dump(
        tiledb_ctx_t* ctx,
        const tiledb_kv_schema_t* kv_schema,
        FILE* out)

    # Key / Value Item
    int tiledb_kv_item_create(
        tiledb_ctx_t* ctx,
        tiledb_kv_item_t** kv_item)

    int tiledb_kv_item_free(
        tiledb_ctx_t* ctx,
        tiledb_kv_item_t** kv_item)

    int tiledb_kv_item_set_key(
        tiledb_ctx_t* ctx,
        tiledb_kv_item_t* kv_item,
        const void* key,
        tiledb_datatype_t key_type,
        uint64_t key_size)

    int tiledb_kv_item_set_value(
        tiledb_ctx_t* ctx,
        tiledb_kv_item_t* kv_item,
        const char* attribute,
        const void* value,
        tiledb_datatype_t value_type,
        uint64_t value_size)

    int tiledb_kv_item_get_key(
        tiledb_ctx_t* ctx,
        tiledb_kv_item_t* kv_item,
        const void** key,
        tiledb_datatype_t* key_type,
        uint64_t* key_size)

    int tiledb_kv_item_get_value(
        tiledb_ctx_t* ctx,
        tiledb_kv_item_t* kv_item,
        const char* attribute,
        const void** value,
        tiledb_datatype_t* value_type,
        uint64_t* value_size)

    # Key / Value store
    int tiledb_kv_create(
        tiledb_ctx_t* ctx,
        const char* kv_uri,
        const tiledb_kv_schema_t* kv_schema)

    int tiledb_kv_consolidate(
        tiledb_ctx_t* ctx,
        const char* kv_uri) nogil

    int tiledb_kv_set_max_buffered_items(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        uint64_t max_items)

    int tiledb_kv_open(
        tiledb_ctx_t* ctx,
        tiledb_kv_t** kv,
        const char* kv_uri,
        const char** attributes,
        unsigned int attribute_num)

    int tiledb_kv_close(
        tiledb_ctx_t* ctx,
        tiledb_kv_t** kv)

    int tiledb_kv_add_item(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        tiledb_kv_item_t* kv_item)

    int tiledb_kv_flush(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv)

    int tiledb_kv_get_item(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        tiledb_kv_item_t** kv_item,
        const void* key,
        tiledb_datatype_t key_type,
        uint64_t key_size)

    # K/V Iterator
    int tiledb_kv_iter_create(
        tiledb_ctx_t* ctx,
        tiledb_kv_iter_t** kv_iter,
        const char* kv_uri,
        const char** attributes,
        unsigned int attribute_num)

    int tiledb_kv_iter_free(
        tiledb_ctx_t* ctx,
        tiledb_kv_iter_t** kv_iter)

    int tiledb_kv_iter_here(
        tiledb_ctx_t* ctx,
        tiledb_kv_iter_t* kv_iter,
        tiledb_kv_item_t** kv_item)

    int tiledb_kv_iter_next(
        tiledb_ctx_t* ctx,
        tiledb_kv_iter_t* kv_iter)

    int tiledb_kv_iter_done(
        tiledb_ctx_t* ctx,
        tiledb_kv_iter_t* kv_iter,
        int* done)

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
        const char* new_path,
        int force) nogil

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
    int tiledb_vfs_create(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t** vfs,
        tiledb_config_t* config)

    int tiledb_vfs_free(
        tiledb_ctx_t* ctx,
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

    int tiledb_vfs_move(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* old_uri,
        const char* new_uri,
        int force) nogil

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

    int tiledb_vfs_fh_free(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t** fh) nogil

    int tiledb_vfs_fh_is_closed(
        tiledb_ctx_t* ctx,
        tiledb_vfs_fh_t* fh,
        int* is_closed) nogil

    int tiledb_vfs_touch(
        tiledb_ctx_t* ctx,
        tiledb_vfs_t* vfs,
        const char* uri) nogil

    # URI
    int tiledb_uri_to_path(
        tiledb_ctx_t* ctx,
        const char* uri,
        char* path_out,
        unsigned* path_length) nogil
