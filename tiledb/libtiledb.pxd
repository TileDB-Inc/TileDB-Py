from libc.stdio cimport FILE
from libc.stdint cimport uint64_t

cdef extern from "tiledb.h":
    # Constants
    enum: TILEDB_OK
    enum: TILEDB_ERR
    enum: TILEDB_OOM

    enum: TILEDB_VAR_NUM
    unsigned int tiledb_var_num()

    # TILEDB_COORDS
    const char* tiledb_coords()

    # Version
    void tiledb_version(int* major, int* minor, int* rev)

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
        TILEDB_BLOSC
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

    # Types
    ctypedef struct tiledb_ctx_t:
        pass
    ctypedef struct tiledb_config_t:
        pass
    ctypedef struct tiledb_error_t:
        pass
    ctypedef struct tiledb_attribute_t:
        pass
    ctypedef struct tiledb_attribute_iter_t:
        pass
    ctypedef struct tiledb_array_schema_t:
        pass
    ctypedef struct tiledb_dimension_t:
        pass
    ctypedef struct tiledb_dimension_iter_t:
        pass
    ctypedef struct tiledb_domain_t:
        pass
    ctypedef struct tiledb_query_t:
        pass
    ctypedef struct tiledb_kv_t:
        pass
    ctypedef struct tiledb_vfs_t:
        pass

    # Config
    int tiledb_config_create(
            tiledb_config_t** config)

    int tiledb_config_free(
            tiledb_config_t* config)

    int tiledb_config_set(
            tiledb_config_t* config, const char* param, const char* value)

    int tiledb_config_set_from_file(
            tiledb_config_t* config, const char* filename)

    int tiledb_config_unset(
            tiledb_config_t* config, const char* param)

    # Context
    int tiledb_ctx_create(
        tiledb_ctx_t** ctx, tiledb_config_t* config)

    int tiledb_ctx_free(
        tiledb_ctx_t* ctx)

    # Error
    int tiledb_error_last(
        tiledb_ctx_t* ctx,
        tiledb_error_t** err)

    int tiledb_error_message(
        tiledb_ctx_t* ctx,
        tiledb_error_t* err,
        char** msg)

    int tiledb_error_free(
        tiledb_ctx_t* ctx,
        tiledb_error_t* err)

    # Group
    int tiledb_group_create(
        tiledb_ctx_t* ctx,
        const char* group)

    # Attribute
    int tiledb_attribute_create(
        tiledb_ctx_t* ctx,
        tiledb_attribute_t** attr,
        const char* name,
        tiledb_datatype_t atype)

    int tiledb_attribute_free(
        tiledb_ctx_t* ctx,
        tiledb_attribute_t* attr)

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

    int tiledb_attribute_from_index(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        unsigned int index,
        tiledb_attribute_t** attr)

    int tiledb_attribute_from_name(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        const char* name,
        tiledb_attribute_t** attr)

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
        tiledb_domain_t* domain)

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

    int tiledb_dimension_from_index(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        unsigned int index,
        tiledb_dimension_t** dim)

    int tiledb_dimension_from_name(
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
        tiledb_dimension_t* dim)

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

    # Dimension Iterator
    int tiledb_dimension_iter_create(
        tiledb_ctx_t* ctx,
        const tiledb_domain_t* domain,
        tiledb_dimension_iter_t** dim_it)

    int tiledb_dimension_iter_free(
        tiledb_ctx_t* ctx,
        tiledb_dimension_iter_t* dim_it);

    int tiledb_dimension_iter_done(
        tiledb_ctx_t* ctx,
        tiledb_dimension_iter_t* dim_it,
        int* done)

    int tiledb_dimension_iter_next(
        tiledb_ctx_t* ctx,
        tiledb_dimension_iter_t* dim_it)

    int tiledb_dimension_iter_here(
        tiledb_ctx_t* ctx,
        tiledb_dimension_iter_t* dim_it,
        const tiledb_dimension_t** dim)

    int tiledb_dimension_iter_first(
        tiledb_ctx_t* ctx,
        tiledb_dimension_iter_t* dim_it)

    # Array schema
    int tiledb_array_schema_create(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t** array_schema)

    int tiledb_array_schema_free(
        tiledb_ctx_t* ctx,
            tiledb_array_schema_t* array_schema)

    int tiledb_array_schema_add_attribute(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_attribute_t* attr)

    int tiledb_array_schema_set_domain(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_domain_t* domain);

    int tiledb_array_schema_set_as_kv(
        tiledb_ctx_t* ctx, tiledb_array_schema_t* array_schema);

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

    int tiledb_array_schema_set_array_type(
        tiledb_ctx_t* ctx,
        tiledb_array_schema_t* array_schema,
        tiledb_array_type_t array_type);

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
        tiledb_compressor_t* coords_compressor,
        int* coords_compression_level)

    int tiledb_array_schema_get_domain(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_domain_t** domain)

    int tiledb_array_schema_get_as_kv(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        int* is_kv)

    int tiledb_array_schema_get_tile_order(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        tiledb_layout_t* tile_order)

    int tiledb_array_schema_get_num_attributes(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        unsigned int* num_attributes)

    int tiledb_array_schema_dump(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* array_schema,
        FILE* out)

    # Attribute iterator
    int tiledb_attribute_iter_create(
        tiledb_ctx_t* ctx,
        const tiledb_array_schema_t* schema,
        tiledb_attribute_iter_t** attr_it)

    int tiledb_attribute_iter_free(
        tiledb_ctx_t* ctx,
        tiledb_attribute_iter_t* attr_it)

    int tiledb_attribute_iter_done(
        tiledb_ctx_t* ctx,
        tiledb_attribute_iter_t* attr_it,
        int* done)

    int tiledb_attribute_iter_next(
        tiledb_ctx_t* ctx,
        tiledb_attribute_iter_t* attr_it)

    int tiledb_attribute_iter_here(
        tiledb_ctx_t* ctx,
        tiledb_attribute_iter_t* attr_it,
        const tiledb_attribute_t** attr)

    int tiledb_attribute_iter_first(
        tiledb_ctx_t* ctx,
        tiledb_attribute_iter_t* attr_it)

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

    int tiledb_query_set_layout(
        tiledb_ctx_t* ctx, tiledb_query_t* query, tiledb_layout_t layout)

    int tiledb_query_free(
        tiledb_ctx_t* ctx, tiledb_query_t* query)

    int tiledb_query_submit(
        tiledb_ctx_t* ctx, tiledb_query_t* query) nogil

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

    int tiledb_query_set_kv(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        tiledb_kv_t* kv);

    int tiledb_query_set_kv_key(
        tiledb_ctx_t* ctx,
        tiledb_query_t* query,
        const void* key,
        tiledb_datatype_t type,
        uint64_t key_size)

    # Array
    int tiledb_array_create(
        tiledb_ctx_t* ctx,
        const char* uri,
        const tiledb_array_schema_t* array_schema)

    int tiledb_array_consolidate(
        tiledb_ctx_t* ctx,
        const char* array_path);

    # Key / Value store
    int tiledb_kv_create(
        tiledb_ctx_t* ctx,
        tiledb_kv_t** kv,
        unsigned int attribute_num,
        const char** attributes,
        tiledb_datatype_t* types,
        unsigned int* nitems)

    int tiledb_kv_free(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv)

    int tiledb_kv_add_key(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        const void* key,
        tiledb_datatype_t key_type,
        uint64_t key_size);

    int tiledb_kv_add_value(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        unsigned int attribute_idx,
        const void* value)

    int tiledb_kv_add_value_var(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        unsigned int attribute_idx,
        const void* value,
        uint64_t value_size)

    int tiledb_kv_get_key_num(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        uint64_t* num)

    int tiledb_kv_get_value_num(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        unsigned int attribute_idx,
        uint64_t* num)

    int tiledb_kv_get_key(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        uint64_t key_idx,
        void** key,
        tiledb_datatype_t* key_type,
        uint64_t* key_size);

    int tiledb_kv_get_value(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        uint64_t obj_index,
        unsigned int attr_idx,
        void** value)

    int tiledb_kv_get_value_var(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        uint64_t obj_index,
        unsigned int attr_idx,
        void** value,
        uint64_t* value_size);

    int tiledb_kv_set_buffer_size(
        tiledb_ctx_t* ctx,
        tiledb_kv_t* kv,
        uint64_t nbytes);

    # Resource management
    int tiledb_object_type(
        tiledb_ctx_t* ctx, const char* path, tiledb_object_t* otype)

    int tiledb_object_remove(
        tiledb_ctx_t* ctx, const char* path)

    int tiledb_object_move(
        tiledb_ctx_t* ctx, const char* old_path, const char* new_path, int force);

    int tiledb_object_walk(
        tiledb_ctx_t* ctx,
        const char* path,
        tiledb_walk_order_t order,
        int (*callback)(const char*, tiledb_object_t, void*),
        void* data)

    # VFS
    int tiledb_vfs_create(
            tiledb_ctx_t* ctx, tiledb_vfs_t** vfs, tiledb_config_t* config)

    int tiledb_vfs_free(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs)

    int tiledb_vfs_create_bucket(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_remove_bucket(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_is_bucket(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_create_dir(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_is_dir(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_remove_dir(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_is_file(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_remove_file(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_file_size(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri, uint64_t* size)

    int tiledb_vfs_move(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs,
            const char* old_uri, const char* new_uri)

    int tiledb_vfs_read(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs,
            const char* uri, uint64_t offset, void* buffer, uint64_t nbytes)

    int tiledb_vfs_write(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs,
            const char* uri, const void* buffer, uint64_t nbytes)

    int tiledb_vfs_sync(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)

    int tiledb_vfs_supports_fs(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, tiledb_filesystem_t fs, int* supports)

    int tiledb_vfs_touch(
            tiledb_ctx_t* ctx, tiledb_vfs_t* vfs, const char* uri)
