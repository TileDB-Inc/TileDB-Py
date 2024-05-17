TileDB Python API Reference
===========================

Modules
-------

Typical usage of the Python interface to TileDB will use the top-level module ``tiledb``, e.g.

.. code-block:: python

   import tiledb

There is also a submodule ``libtiledb`` which contains the necessary bindings to the underlying TileDB native library. Most of the time you will not need to interact with ``tiledb.libtiledb`` unless you need native-library specific information, e.g. the version number:

.. code-block:: python

   import tiledb
   tiledb.libtiledb.version()  # Native TileDB library version number

Getting Started
---------------

Arrays may be opened with the ``tiledb.open`` function:

.. autofunction:: tiledb.open

Data import helpers
-------------------

.. autofunction:: tiledb.from_numpy
.. autofunction:: tiledb.from_csv
.. autofunction:: tiledb.from_pandas

Context
-------

.. autoclass:: tiledb.Ctx
   :members:

.. autofunction:: tiledb.default_ctx

Config
------

.. autoclass:: tiledb.Config
   :members:

Array Schema
------------

.. autoclass:: tiledb.ArraySchema
   :members:

.. autofunction:: tiledb.empty_like

Attribute
---------

.. autoclass:: tiledb.Attr
   :members:

Filters
-------

.. autoclass:: tiledb.FilterList
   :members:

   .. automethod:: __getitem__(idx)
   .. automethod:: __len__

.. autoclass:: tiledb.CompressionFilter
   :members:
.. autoclass:: tiledb.GzipFilter
   :members:
.. autoclass:: tiledb.ZstdFilter
   :members:
.. autoclass:: tiledb.LZ4Filter
   :members:
.. autoclass:: tiledb.Bzip2Filter
   :members:
.. autoclass:: tiledb.RleFilter
   :members:
.. autoclass:: tiledb.DoubleDeltaFilter
   :members:
.. autoclass:: tiledb.BitShuffleFilter
   :members:
.. autoclass:: tiledb.ByteShuffleFilter
   :members:
.. autoclass:: tiledb.BitWidthReductionFilter
   :members:
.. autoclass:: tiledb.PositiveDeltaFilter
   :members:

Dimension
---------

.. autoclass:: tiledb.Dim
   :members:

Domain
------

.. autoclass:: tiledb.Domain
   :members:

Array
-----

.. autoclass:: tiledb.libtiledb.Array
   :members:

.. autofunction:: tiledb.consolidate
.. autofunction:: tiledb.vacuum

Dense Array
-----------

.. autoclass:: tiledb.DenseArray
   :members: query
   :special-members: __getitem__, __setitem__

Sparse Array
------------

.. autoclass:: tiledb.SparseArray
   :members: query
   :special-members: __getitem__, __setitem__

Query
---------------

.. autoclass:: tiledb.libtiledb.Query
   :members:

Query Condition
---------------

.. autoclass:: tiledb.QueryCondition
   :members:

Group
-----

.. autoclass:: tiledb.Group
   :members:

   .. automethod:: __getitem__(member)
   .. automethod:: __delitem__(uri)
   .. automethod:: __contains__(member)
   .. automethod:: __len__

.. autoclass:: tiledb.Group.GroupMetadata
   :members:

   .. automethod:: __setitem__(key, value)
   .. automethod:: __getitem__(key)
   .. automethod:: __delitem__(key)
   .. automethod:: __contains__(key)
   .. automethod:: __len__
   .. automethod:: __keys__
   .. automethod:: __values__
   .. automethod:: __items__

Object
------

.. autoclass:: tiledb.Object

   .. autoattribute:: uri
   .. autoattribute:: type

Object Management
-----------------

.. autofunction:: tiledb.array_exists
.. autofunction:: tiledb.group_create
.. autofunction:: tiledb.object_type
.. autofunction:: tiledb.remove
.. autofunction:: tiledb.move
.. autofunction:: tiledb.ls
.. autofunction:: tiledb.walk

Fragment Info
-------------

.. autoclass:: tiledb.FragmentInfoList
   :members:
.. autoclass:: tiledb.FragmentInfo
   :members:

Exceptions
----------

.. autoexception:: tiledb.TileDBError
   :members:


VFS
---

.. autoclass:: tiledb.VFS
   :members:

.. autoclass:: tiledb.FileIO
   :members:

   .. automethod:: __len__

Filestore
---------

.. autoclass:: tiledb.Filestore
   :members:

   .. automethod:: __len__

Version
-------

.. autofunction:: tiledb.libtiledb.version

Statistics
----------

.. autofunction:: tiledb.stats_enable
.. autofunction:: tiledb.stats_disable
.. autofunction:: tiledb.stats_reset
.. autofunction:: tiledb.stats_dump
