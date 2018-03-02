TileDB Python API Reference
===========================

Modules
-------

``import tiledb``


Exceptions
----------

.. autoexception:: tiledb.TileDBError
   :members:

Classes
-------

.. autoclass:: tiledb.Ctx
   :members:

.. autoclass:: tiledb.Config
   :members:

.. autoclass:: tiledb.Dim
   :members:

.. autoclass:: tiledb.Domain
   :members:

.. autoclass:: tiledb.Attr
   :members:

.. autoclass:: tiledb.KV
   :members:

.. autoclass:: tiledb.VFS

Array
-----

.. autoclass:: tiledb.ArraySchema
   :members:

.. autoclass:: tiledb.DenseArray
   :members:
   
   .. automethod:: __getitem__(selection)
   .. automethod:: __setitem__(selection, value)
    
.. autoclass:: tiledb.SparseArray
   :members:
   
   .. automethod:: __getitem__(selection)
   .. automethod:: __setitem__(selection, value)


.. autofunction:: tiledb.array_consolidate

Resource Management
-------------------

.. autofunction:: tiledb.group_create
.. autofunction:: tiledb.object_type
.. autofunction:: tiledb.move
.. autofunction:: tiledb.remove
.. autofunction:: tiledb.ls
.. autofunction:: tiledb.walk

Version
-------

.. autofunction:: tiledb.libtiledb.version
