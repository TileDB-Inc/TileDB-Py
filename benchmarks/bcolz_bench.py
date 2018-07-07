import bcolz

import numpy as np
import sys
import os
import tiledb

from subprocess import call

# Name of the array to create.
tiledb_array_name = "test_tiledb"
bcolz_array_name = "test_bcolz"

TEST_ARRAY = np.random.rand(100000000)

MB = TEST_ARRAY.nbytes / (1024 * 1024)
print("With random float64 array of: {} MB".format(MB))

# Create a TileDB context
ctx = tiledb.Ctx()

# Check if the array already exists.
if tiledb.object_type(ctx, tiledb_array_name) == "array":
    import shutil
    shutil.rmtree("./".join((tiledb_array_name,)))

if os.path.exists(bcolz_array_name):
    import shutil
    shutil.rmtree("./".join((bcolz_array_name,)))

def create_array():

    # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4].
    dom = tiledb.Domain(ctx, 
            tiledb.Dim(ctx, name="rows", domain=(0, (TEST_ARRAY.shape[0] - 1)), 
                       tile=10000, dtype=np.int32))

    # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=False,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.float64)])

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(tiledb_array_name, schema)

def write_array():
    # Open the array and write to it.
    with tiledb.DenseArray(ctx, tiledb_array_name, mode='w') as A:
        A[:] = TEST_ARRAY


def read_array():
    # Open the array and read from it.
    with tiledb.DenseArray(ctx, tiledb_array_name, mode='r') as A:
        # Slice only rows 1, 2 and cols 2, 3, 4.
        data = A[0:(TEST_ARRAY.size // 2)]

def write_bcolz():
   cparams = bcolz.cparams(0)
   A = bcolz.carray(TEST_ARRAY, rootdir=bcolz_array_name, mode='w')
   A.flush()
   call(["sudo", "sync"])

def read_bcolz():
    A = bcolz.carray(rootdir=bcolz_array_name, mode='r')
    data = A[0:(TEST_ARRAY.size // 2)]

import time

start = time.time()
create_array()
end = time.time()
print("CREATING a tiledb array took {:.3f}".format(end - start))
call(["sudo", "sync"])
call(["sudo", "purge"])

start = time.time()
write_array()
end = time.time()
print("WRITING a tiledb array took {:.3f}".format(end - start))
print("    - {:.1f} MB/s\n".format(MB / (end - start)))
call(["sudo", "sync"])
call(["sudo", "purge"])

start = time.time()
read_array()
end = time.time()
print("READING a tiledb array took {:.3f}".format(end - start))
print("    - {:.1f} MB/s\n".format((MB //2) / (end - start)))
call(["sudo", "purge"])

start = time.time()
write_bcolz()
end = time.time()
print("WRITING a bcolz array took {:.3f}".format(end - start))
print("    - {:.1f} MB/s\n".format(MB / (end - start)))
call(["sudo", "sync"])
call(["sudo", "purge"])

start = time.time()
read_bcolz()
end = time.time()
print("READING a bcolz array took {:.3f}".format(end - start))
print("    - {:.1f} MB/s\n".format((MB //2) / (end - start)))
call(["sudo", "sync"])
call(["sudo", "purge"])
