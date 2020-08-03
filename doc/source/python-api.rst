TileDB Python API Reference
===========================

.. warning::

   The Python interface to TileDB is still under development and the API is subject to change.

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
.. autoclass:: tiledb.libtiledb.CompressionFilter
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
   :members:
   
   .. automethod:: __getitem__(selection)
   .. automethod:: __setitem__(selection, value)
   .. automethod:: from_numpy(uri, array, ctx=None, **kwargs)

Sparse Array
------------

.. autoclass:: tiledb.SparseArray
   :members:
   
   .. automethod:: __getitem__(selection)
   .. automethod:: __setitem__(selection, value)

Object Management
-----------------

.. autofunction:: tiledb.array_exists
.. autofunction:: tiledb.group_create
.. autofunction:: tiledb.object_type
.. autofunction:: tiledb.remove
.. autofunction:: tiledb.move
.. autofunction:: tiledb.ls
.. autofunction:: tiledb.walk

Exceptions
----------

.. autoexception:: tiledb.TileDBError
   :members:


VFS
---

.. autoclass:: tiledb.VFS
   :members:

Version
-------

.. autofunction:: tiledb.libtiledb.version

Statistics
----------

.. autofunction:: tiledb.stats_enable
.. autofunction:: tiledb.stats_disable
.. autofunction:: tiledb.stats_reset
.. autofunction:: tiledb.stats_dump
