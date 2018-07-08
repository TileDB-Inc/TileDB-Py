TileDB Python API Reference
===========================

.. warning::

   The Python interface to TileDB is still under development and the API is subject to change.

Modules
-------

Typical usage of the Python interace to TileDB will use the top-level module ``tiledb``, e.g.

.. code-block:: python

   import tiledb

There is also a submodule ``libtiledb`` which contains the necessary bindings to the underlying TileDB native library. Most of the time you will not need to interact with ``tiledb.libtiledb`` unless you need native-library specific information, e.g. the version number:

.. code-block:: python

   import tiledb
   tiledb.libtiledb.version()  # Native TileDB library version number

Exceptions
----------

.. autoexception:: tiledb.TileDBError
   :members:

Context
-------

.. autoclass:: tiledb.Ctx
   :members:

Config
------

.. autoclass:: tiledb.Config
   :members:

Array Schema
------------

.. autoclass:: tiledb.ArraySchema
   :members:

Attribute
---------

.. autoclass:: tiledb.Attr
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

Dense Array
-----------

.. autoclass:: tiledb.DenseArray
   :members:
   
   .. automethod:: __getitem__(selection)
   .. automethod:: __setitem__(selection, value)

Sparse Array
------------

.. autoclass:: tiledb.SparseArray
   :members:
   
   .. automethod:: __getitem__(selection)
   .. automethod:: __setitem__(selection, value)

Key-value Schema
----------------

.. autoclass:: tiledb.KVSchema
   :members:

Key-value
---------

.. autoclass:: tiledb.KV
   :members:

Object Management
-----------------

.. autofunction:: tiledb.group_create
.. autofunction:: tiledb.object_type
.. autofunction:: tiledb.remove
.. autofunction:: tiledb.move
.. autofunction:: tiledb.ls
.. autofunction:: tiledb.walk

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