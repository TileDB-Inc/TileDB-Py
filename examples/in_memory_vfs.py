# in_memory_vfs.py
#
# LICENSE
#
# The MIT License
#
# Copyright (c) 2024 TileDB, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# DESCRIPTION
#
# When run, this program will create a random numpy array, create a TileDB
# DenseArray from it in memory, and read the entire array back from memory.
# It then compares the original numpy array with the TileDB array to check
# that data was written and read correctly (prints True) or not (prints False).
#

import numpy as np

import tiledb

# Create TileDB VFS
vfs = tiledb.VFS()

file = "mem://myarray"
data = np.random.rand(10, 10)

if not vfs.is_file(file):
    with tiledb.from_numpy(file, data):
        pass

with tiledb.open(file) as A:
    print(np.all(A[:] == data))
