import os
import pprint
import warnings

import tiledb

from .main import PyFragmentInfo

"""
Classes and functions relating to TileDB fragments.
"""


class FragmentInfoList:
    """
    Class representing an ordered list of FragmentInfo objects.

    :param array_uri: URI for the TileDB array (any supported TileDB URI)
    :type array_uri: str
    :param include_mbrs: (default False) include minimum bounding rectangles in FragmentInfo result
    :type include_mbrs: bool
    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx

    :ivar uri: URIs of fragments
    :ivar version: Fragment version of each fragment
    :ivar nonempty_domain: Non-empty domain of each fragment
    :ivar cell_num: Number of cells in each fragment
    :ivar timestamp_range: Timestamp range of when each fragment was written
    :ivar sparse: For each fragment, True if fragment is sparse, else False
    :ivar has_consolidated_metadata: For each fragment, True if fragment has consolidated fragment metadata, else False
    :ivar unconsolidated_metadata_num: Number of unconsolidated metadata fragments in each fragment
    :ivar to_vacuum: URIs of already consolidated fragments to vacuum
    :ivar mbrs: (TileDB Embedded 2.5.0+ only) The mimimum bounding rectangle of each fragment; only present when `include_mbrs=True`
    :ivar array_schema_name: (TileDB Embedded 2.5.0+ only) The array schema's name

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4] and space tiles 2x2
    ...     dom = tiledb.Domain(
    ...         tiledb.Dim(name="rows", domain=(1, 4), tile=2, dtype=np.int32),
    ...         tiledb.Dim(name="cols", domain=(1, 4), tile=2, dtype=np.int32),
    ...     )
    ...     # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
    ...     schema = tiledb.ArraySchema(
    ...         domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype=np.int32)]
    ...     )
    ...     # Set URI of the array
    ...     uri = tmp + "/array"
    ...     # Create the (empty) array on disk.
    ...     tiledb.Array.create(uri, schema)
    ...
    ...     # Write three fragments to the array
    ...     with tiledb.DenseArray(uri, mode="w") as A:
    ...         A[1:3, 1:5] = np.array(([[1, 2, 3, 4], [5, 6, 7, 8]]))
    ...     with tiledb.DenseArray(uri, mode="w") as A:
    ...         A[2:4, 2:4] = np.array(([101, 102], [103, 104]))
    ...     with tiledb.DenseArray(uri, mode="w") as A:
    ...         A[3:4, 4:5] = np.array(([202]))
    ...
    ...     # tiledb.array_fragments() requires TileDB-Py version > 0.8.5
    ...     fragments_info = tiledb.array_fragments(uri)
    ...
    ...     "====== FRAGMENTS  INFO ======"
    ...     f"number of fragments: {len(fragments_info)}"
    ...     f"nonempty domains: {fragments_info.nonempty_domain}"
    ...     f"sparse fragments: {fragments_info.sparse}"
    ...
    ...     for fragment in fragments_info:
    ...         f"===== FRAGMENT NUMBER {fragment.num} ====="
    ...         f"is sparse: {fragment.sparse}"
    ...         f"cell num: {fragment.cell_num}"
    ...         f"has consolidated metadata: {fragment.has_consolidated_metadata}"
    ...         f"nonempty domain: {fragment.nonempty_domain}"
    '====== FRAGMENTS  INFO ======'
    'number of fragments: 3'
    'nonempty domains: (((1, 2), (1, 4)), ((2, 3), (2, 3)), ((3, 3), (4, 4)))'
    'sparse fragments: (False, False, False)'
    '===== FRAGMENT NUMBER 0 ====='
    'is sparse: False'
    'cell num: 8'
    'has consolidated metadata: False'
    'nonempty domain: ((1, 2), (1, 4))'
    '===== FRAGMENT NUMBER 1 ====='
    'is sparse: False'
    'cell num: 16'
    'has consolidated metadata: False'
    'nonempty domain: ((2, 3), (2, 3))'
    '===== FRAGMENT NUMBER 2 ====='
    'is sparse: False'
    'cell num: 4'
    'has consolidated metadata: False'
    'nonempty domain: ((3, 3), (4, 4))'

    """

    def __init__(self, array_uri, include_mbrs=False, ctx=None):
        if ctx is None:
            ctx = tiledb.default_ctx()

        schema = tiledb.ArraySchema.load(array_uri, ctx=ctx)

        self.array_uri = array_uri

        fi = PyFragmentInfo(self.array_uri, schema, include_mbrs, ctx)

        self.__nums = fi.get_num_fragments()
        self.uri = fi.get_uri()
        self.version = fi.get_version()
        self.nonempty_domain = fi.get_nonempty_domain()
        self.cell_num = fi.get_cell_num()
        self.timestamp_range = fi.get_timestamp_range()
        self.sparse = fi.get_sparse()
        self.unconsolidated_metadata_num = fi.get_unconsolidated_metadata_num()
        self.has_consolidated_metadata = fi.get_has_consolidated_metadata()
        self.to_vacuum = fi.get_to_vacuum()

        if include_mbrs:
            if tiledb.libtiledb.version() >= (2, 5, 0):
                self.mbrs = fi.get_mbrs()
            else:
                warnings.warn(
                    "MBRs for fragments not available; "
                    "please install libtiledb 2.5.0+",
                    UserWarning,
                )

        if tiledb.libtiledb.version() >= (2, 5, 0):
            self.array_schema_name = fi.get_array_schema_name()

    @property
    def non_empty_domain(self):
        raise tiledb.TileDBError(
            "FragmentInfoList.non_empty_domain is deprecated; "
            "you must use FragmentInfoList.nonempty_domain",
            "This message will be removed in 0.21.0.",
        )

    @property
    def to_vacuum_num(self):
        raise tiledb.TileDBError(
            "FragmentInfoList.to_vacuum_num is deprecated; "
            "you must use len(FragmentInfoList.to_vacuum)",
            "This message will be removed in 0.21.0.",
        )

    @property
    def to_vacuum_uri(self):
        raise tiledb.TileDBError(
            "FragmentInfoList.to_vacuum_uri is deprecated; "
            "you must use FragmentInfoList.to_vacuum",
            "This message will be removed in 0.21.0.",
        )

    @property
    def dense(self):
        raise tiledb.TileDBError(
            "FragmentInfoList.dense is deprecated; "
            "you must use FragmentInfoList.sparse",
            "This message will be removed in 0.21.0.",
        )

    def __getattr__(self, name):
        if name == "mbrs":
            raise AttributeError(
                "'FragmentInfoList' object has no attribute 'mbrs'. "
                "(Hint: retrieving minimum bounding rectangles is disabled "
                "by default to optimize speed and space. "
                "Use tiledb.array_fragments(include_mbrs=True) to enable)"
            )
        return self.__getattribute__(name)

    def __iter__(self):
        return FragmentsInfoIterator(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [FragmentInfo(self, idx) for idx in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            return FragmentInfo(self, key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return self.__nums

    def __repr__(self):
        public_attrs = {
            key: value
            for (key, value) in self.__dict__.items()
            if not key.startswith("_")
        }
        return pprint.PrettyPrinter().pformat(public_attrs)

    def _repr_html_(self) -> str:
        from io import StringIO

        output = StringIO()
        output.write("<section>\n")
        output.write(f"<h2>Fragments for {self.array_uri}</h2>\n")
        for frag in self:
            output.write("<details>\n")
            output.write(f"<summary>{frag.uri}</summary>\n")
            output.write(frag._repr_html_())
            output.write("</details>\n")
        output.write("</section>\n")
        return output.getvalue()


class FragmentsInfoIterator:
    """
    Iterator class for the FragmentsInfo container.
    """

    def __init__(self, fragments):
        self._fragments = fragments
        self._index = 0

    def __next__(self):
        if self._index < len(self._fragments):
            fi = FragmentInfo(self._fragments, self._index)
            self._index += 1
            return fi
        raise StopIteration


class FragmentInfo:
    """
    Class representing the metadata for a single fragment. See :py:class:`tiledb.FragmentInfoList` for example of usage.

    :ivar uri: URIs of fragments
    :ivar version: Fragment version of each fragment
    :ivar nonempty_domain: Non-empty domain of each fragment
    :ivar cell_num: Number of cells in each fragment
    :ivar timestamp_range: Timestamp range of when each fragment was written
    :ivar sparse: For each fragment, True if fragment is sparse, else False
    :ivar has_consolidated_metadata: For each fragment, True if fragment has consolidated fragment metadata, else False
    :ivar unconsolidated_metadata_num: Number of unconsolidated metadata fragments in each fragment
    :ivar to_vacuum: URIs of already consolidated fragments to vacuum
    :ivar mbrs: (TileDB Embedded 2.5.0+ only) The mimimum bounding rectangle of each fragment; only present when `include_mbrs=True`
    :ivar array_schema_name: (TileDB Embedded 2.5.0+ only) The array schema's name
    """

    def __init__(self, fragments: FragmentInfoList, num):
        self.num = num
        self.uri = fragments.uri[num]
        self.version = fragments.version[num]
        self.nonempty_domain = fragments.nonempty_domain[num]
        self.cell_num = fragments.cell_num[num]
        self.timestamp_range = fragments.timestamp_range[num]
        self.sparse = fragments.sparse[num]
        self.has_consolidated_metadata = fragments.has_consolidated_metadata[num]
        self.unconsolidated_metadata_num = fragments.unconsolidated_metadata_num

        if hasattr(fragments, "mbrs"):
            self.mbrs = fragments.mbrs[num]

        if hasattr(fragments, "array_schema_name"):
            self.array_schema_name = fragments.array_schema_name[num]

    def __repr__(self):
        public_attrs = {
            key: value
            for (key, value) in self.__dict__.items()
            if not key.startswith("_")
        }
        return pprint.PrettyPrinter().pformat(public_attrs)

    def _repr_html_(self) -> str:
        from io import StringIO

        output = StringIO()
        output.write("<section>\n")
        output.write("<table>\n")
        for key in self.__dict__:
            if not key.startswith("_"):
                output.write("<tr>\n")
                output.write(f"<td>{key}</td>\n")
                output.write(f"<td>{self.__dict__[key]}</td>\n")
                output.write("</tr>\n")
        output.write("</table>\n")
        output.write("</section>\n")

        return output.getvalue()

    def __getattr__(self, name):
        if name == "mbrs":
            raise AttributeError(
                "'FragmentInfo' object has no attribute 'mbrs'. "
                "(Hint: retrieving minimum bounding rectangles is disabled "
                "by default to optimize speed and space. "
                "Use tiledb.array_fragments(include_mbrs=True) to enable)"
            )
        return self.__getattribute__(name)

    @property
    def non_empty_domain(self):
        raise tiledb.TileDBError(
            "FragmentInfo.non_empty_domain is deprecated; "
            "you must use FragmentInfo.nonempty_domain. ",
            "This message will be removed in 0.21.0.",
        )

    @property
    def to_vacuum_num(self):
        raise tiledb.TileDBError(
            "FragmentInfo.to_vacuum_num is deprecated; "
            "you must use len(FragmentInfoList.to_vacuum).",
            "This message will be removed in 0.21.0.",
        )

    @property
    def to_vacuum_uri(self):
        raise tiledb.TileDBError(
            "FragmentInfo.to_vacuum_uri is deprecated; "
            "you must use FragmentInfoList.to_vacuum.",
            "This message will be removed in 0.21.0.",
        )

    @property
    def dense(self):
        raise tiledb.TileDBError(
            "FragmentInfo.dense is deprecated; you must use FragmentInfo.sparse",
            "This message will be removed in 0.21.0.",
        )


def FragmentsInfo(array_uri, ctx=None):
    """
    Deprecated in 0.8.8.

    Renamed to FragmentInfoList to make name more distinguishable from FragmentInfo.
    """

    raise tiledb.TileDBError(
        "FragmentsInfo is deprecated; you must use FragmentInfoList. "
        "This message will be removed in 0.21.0.",
    )


def delete_fragments(
    uri, timestamp_range, config=None, ctx=None, verbose=False, dry_run=False
):
    """
    Delete fragments from an array located at uri that falls within a given
    timestamp_range.

    :param str uri: URI for the TileDB array (any supported TileDB URI)
    :param (int, int) timestamp_range: (default None) If not None, vacuum the
        array using the given range (inclusive)
    :param config: Override the context configuration. Defaults to ctx.config()
    :param ctx: (optional) TileDB Ctx
    :param verbose: (optional) Print fragments being deleted (default: False)
    :param dry_run: (optional) Preview fragments to be deleted without
        running (default: False)
    """
    raise tiledb.TileDBError(
        "tiledb.delete_fragments is deprecated; you must use Array.delete_fragments. "
        "This message will be removed in 0.21.0."
    )


def create_array_from_fragments(
    src_uri,
    dst_uri,
    timestamp_range,
    config=None,
    ctx=None,
    verbose=False,
    dry_run=False,
):
    """
    (POSIX only). Create a new array from an already existing array by selecting
    fragments that fall withing a given timestamp_range. The original array is located
    at src_uri and the new array is created at dst_uri.

    :param str src_uri: URI for the source TileDB array (any supported TileDB URI)
    :param str dst_uri: URI for the newly created TileDB array (any supported TileDB URI)
    :param (int, int) timestamp_range: (default None) If not None, vacuum the
        array using the given range (inclusive)
    :param config: Override the context configuration. Defaults to ctx.config()
    :param ctx: (optional) TileDB Ctx
    :param verbose: (optional) Print fragments being copied (default: False)
    :param dry_run: (optional) Preview fragments to be copied without
        running (default: False)
    """
    if tiledb.array_exists(dst_uri):
        raise tiledb.TileDBError(f"Array URI `{dst_uri}` already exists")

    if not isinstance(timestamp_range, tuple) and len(timestamp_range) != 2:
        raise TypeError(
            "'timestamp_range' argument expects tuple(start: int, end: int)"
        )

    if not ctx:
        ctx = tiledb.default_ctx()

    if config is None:
        config = tiledb.Config(ctx.config())

    vfs = tiledb.VFS(config=config, ctx=ctx)

    fragment_info = tiledb.array_fragments(src_uri)

    if len(fragment_info) < 1:
        print("Cannot create new array; no fragments to copy")
        return

    if verbose or dry_run:
        print(f"Creating directory for array at {dst_uri}\n")

    if not dry_run:
        vfs.create_dir(dst_uri)

    src_lock = os.path.join(src_uri, "__lock.tdb")
    dst_lock = os.path.join(dst_uri, "__lock.tdb")

    if verbose or dry_run:
        print(f"Copying lock file {dst_uri}\n")

    if not dry_run:
        vfs.copy_file(f"{src_lock}", f"{dst_lock}")

    list_new_style_schema = [ver >= 10 for ver in fragment_info.version]
    is_mixed_versions = len(set(list_new_style_schema)) > 1
    if is_mixed_versions:
        raise tiledb.TileDBError(
            "Cannot copy fragments - this array contains a mix of old and "
            "new style schemas"
        )
    is_new_style_schema = list_new_style_schema[0]

    for frag in fragment_info:
        if not (
            timestamp_range[0] <= frag.timestamp_range[0]
            and frag.timestamp_range[1] <= timestamp_range[1]
        ):
            continue

        schema_name = frag.array_schema_name
        if is_new_style_schema:
            schema_name = os.path.join("__schema", schema_name)
        src_schema = os.path.join(src_uri, schema_name)
        dst_schema = os.path.join(dst_uri, schema_name)

        if verbose or dry_run:
            print(f"Copying schema `{src_schema}` to `{dst_schema}`\n")

        if not dry_run:
            if is_new_style_schema:
                new_style_schema_uri = os.path.join(dst_uri, "__schema")
                if not vfs.is_dir(new_style_schema_uri):
                    vfs.create_dir(new_style_schema_uri)

            if not vfs.is_file(dst_schema):
                vfs.copy_file(src_schema, dst_schema)

        base_name = os.path.basename(frag.uri)
        if frag.version < 12:
            frag_name = base_name
        else:
            vfs.create_dir(os.path.join(dst_uri, "__fragments"))
            frag_name = os.path.join("__fragments", base_name)

        src_frag = os.path.join(src_uri, frag_name)
        dst_frag = os.path.join(dst_uri, frag_name)

        if frag.version < 12:
            ok_or_wrt_name = f"{base_name}.ok"
        else:
            vfs.create_dir(os.path.join(dst_uri, "__commits"))
            ok_or_wrt_name = os.path.join("__commits", f"{base_name}.wrt")

        src_ok_or_wrt = os.path.join(src_uri, ok_or_wrt_name)
        dst_ok_or_wrt = os.path.join(dst_uri, ok_or_wrt_name)

        if verbose or dry_run:
            print(f"Copying `{src_frag}` to `{dst_frag}`\n")
            print(f"Copying `{src_ok_or_wrt}` to `{dst_ok_or_wrt}`\n")

        if not dry_run:
            vfs.copy_dir(src_frag, dst_frag)
            vfs.copy_file(src_ok_or_wrt, dst_ok_or_wrt)


def copy_fragments_to_existing_array(
    src_uri,
    dst_uri,
    timestamp_range,
    config=None,
    ctx=None,
    verbose=False,
    dry_run=False,
):
    """
    (POSIX only). Copy fragments from an array at src_uri to another array at
    dst_uri by selecting fragments that fall withing a given timestamp_range.

    :param str src_uri: URI for the source TileDB array (any supported TileDB URI)
    :param str dst_uri: URI for the destination TileDB array (any supported TileDB URI)
    :param (int, int) timestamp_range: (default None) If not None, vacuum the
        array using the given range (inclusive)
    :param config: Override the context configuration. Defaults to ctx.config()
    :param ctx: (optional) TileDB Ctx
    :param verbose: (optional) Print fragments being copied (default: False)
    :param dry_run: (optional) Preview fragments to be copied without
        running (default: False)
    """
    if not tiledb.array_exists(dst_uri):
        raise tiledb.TileDBError(f"Array URI `{dst_uri}` does not exist")

    if not isinstance(timestamp_range, tuple) and len(timestamp_range) != 2:
        raise TypeError(
            "'timestamp_range' argument expects tuple(start: int, end: int)"
        )

    if not ctx:
        ctx = tiledb.default_ctx()

    if config is None:
        config = tiledb.Config(ctx.config())

    vfs = tiledb.VFS(config=config, ctx=ctx)

    dst_schema_file = os.path.join(dst_uri, "__array_schema.tdb")
    src_schema_file = os.path.join(src_uri, "__array_schema.tdb")
    dst_schema_dir = os.path.join(dst_uri, "__schema")
    src_schema_dir = os.path.join(src_uri, "__schema")

    is_old_style = vfs.is_file(dst_schema_file) and vfs.is_file(src_schema_file)
    is_new_style = vfs.is_dir(dst_schema_dir) and vfs.is_dir(src_schema_dir)

    if is_old_style and is_new_style:
        raise tiledb.TileDBError(
            "Mix of old and new style schemas detected. There can only be "
            "one schema version present in both the source and destination "
            "arrays and both must be identical"
        )
    elif is_new_style:

        def filtered_schema_dir(uri):
            return [x for x in vfs.ls(uri) if "__enumerations" not in x]

        if (
            len(filtered_schema_dir(dst_schema_dir)) != 1
            or len(filtered_schema_dir(src_schema_dir)) != 1
        ):
            raise tiledb.TileDBError(
                "Mutltiple evolved schemas detected. There can only be one "
                "schema version present in both the source and destination "
                "arrays and both must be identical"
            )
        schema_name = os.path.basename(vfs.ls(src_schema_dir)[0])
        src_schema = os.path.join(src_uri, "__schema", schema_name)
        dst_schema = os.path.join(dst_uri, "__schema", schema_name)

    if tiledb.ArraySchema.load(src_uri) != tiledb.ArraySchema.load(dst_uri):
        raise tiledb.TileDBError(
            "The source and destination array must have matching schemas."
        )

    if is_new_style:
        if verbose or dry_run:
            print(f"Copying schema `{src_schema}` to `{dst_schema}`\n")

        if not dry_run:
            vfs.copy_file(src_schema, dst_schema)

    array_fragments = tiledb.array_fragments(src_uri)

    for frag in array_fragments:
        if not (
            timestamp_range[0] <= frag.timestamp_range[0]
            and frag.timestamp_range[1] <= timestamp_range[1]
        ):
            continue

        base_name = os.path.basename(frag.uri)
        if frag.version < 12:
            frag_name = base_name
        else:
            vfs.create_dir(os.path.join(dst_uri, "__fragments"))
            frag_name = os.path.join("__fragments", base_name)

        src_frag = os.path.join(src_uri, frag_name)
        dst_frag = os.path.join(dst_uri, frag_name)

        if frag.version < 12:
            ok_or_wrt_name = f"{base_name}.ok"
        else:
            ok_or_wrt_name = os.path.join("__commits", f"{base_name}.wrt")

        src_ok_or_wrt = os.path.join(src_uri, ok_or_wrt_name)
        dst_ok_or_wrt = os.path.join(dst_uri, ok_or_wrt_name)

        if src_frag == dst_frag:
            if verbose or dry_run:
                print(
                    f"Fragment {src_frag} not copied. Already exists in "
                    "destination array.\n"
                )
            continue

        if verbose or dry_run:
            print(f"Copying `{src_frag}` to `{dst_frag}`\n")
            print(f"Copying `{src_ok_or_wrt}` to `{dst_ok_or_wrt}`\n")

        if not dry_run:
            vfs.copy_dir(src_frag, dst_frag)
            vfs.copy_file(src_ok_or_wrt, dst_ok_or_wrt)
