import os
import time

import numpy as np
from numpy.testing import assert_array_equal

import tiledb.cc as lt


def test_group_metadata(tmp_path):
    int_data = np.array([1, 2, 3])
    flt_data = np.array([1.5, 2.5, 3.5])

    ctx = lt.Context()

    grp_path = os.path.join(tmp_path, "test_groups_metadata")
    lt.Group._create(ctx, grp_path)
    grp = lt.Group(ctx, grp_path, lt.QueryType.WRITE)
    grp._put_metadata("int", int_data)
    grp._put_metadata("flt", flt_data)
    grp._close()

    grp._open(lt.QueryType.READ)
    assert grp._metadata_num() == 2
    assert grp._has_metadata("int")
    assert_array_equal(grp._get_metadata("int")[0], int_data)
    assert grp._has_metadata("flt")
    assert_array_equal(grp._get_metadata("flt")[0], flt_data)
    grp._close()

    time.sleep(0.001)
    grp._open(lt.QueryType.WRITE)
    grp._delete_metadata("int")
    grp._close()

    grp = lt.Group(ctx, grp_path, lt.QueryType.READ)
    assert grp._metadata_num() == 1
    assert not grp._has_metadata("int")
    grp._close()


def test_group_members(tmp_path):
    ctx = lt.Context()

    grp_path = os.path.join(tmp_path, "test_group_metadata")
    lt.Group._create(ctx, grp_path)
    grp = lt.Group(ctx, grp_path, lt.QueryType.WRITE)

    subgrp_path = os.path.join(tmp_path, "test_group_0")
    lt.Group._create(ctx, subgrp_path)
    grp._add(subgrp_path)
    grp._close()

    grp._open(lt.QueryType.READ)
    assert grp._member_count() == 1
    member = grp._member(0)
    assert os.path.basename(member._uri) == os.path.basename(subgrp_path)
    assert member._type == lt.ObjectType.GROUP
    grp._close()

    grp._open(lt.QueryType.WRITE)
    grp._remove(subgrp_path)
    grp._close()

    grp = lt.Group(ctx, grp_path, lt.QueryType.READ)
    assert grp._member_count() == 0
    grp._close()
