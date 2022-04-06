import numpy as np
from numpy.testing import assert_array_equal
import pytest

import os

from tiledb import cc as lt


def test_group_metadata(tmp_path):
    int_data = np.array([1, 2, 3])
    flt_data = np.array([1.5, 2.5, 3.5])

    ctx = lt.Context()

    grp_path = os.path.join(tmp_path, "test_groups_metadata")
    lt.Group._create(ctx, grp_path)
    grp = lt.Group(ctx, grp_path, lt.QueryType.WRITE)
    grp._put_metadata("int", int_data)
    grp._put_metadata("flt", flt_data)
    grp.close()

    grp._open(lt.QueryType.READ)
    assert grp._metadata_num() == 2
    assert grp._has_metadata("int")
    assert_array_equal(grp._get_metadata("int"), int_data)
    assert grp._has_metadata("flt")
    assert_array_equal(grp._get_metadata("flt"), flt_data)
    grp.close()

    # NOTE uncomment when deleting is "synced" in core
    # grp.open(lt.QueryType.WRITE)
    # grp._delete_metadata("int")
    # grp.close()

    # grp = lt.Group(ctx, grp_path, lt.QueryType.READ)
    # assert grp.metadata_num() == 1
    # assert not grp._has_metadata("int")
    # grp.close()


def test_group_members(tmp_path):
    ctx = lt.Context()

    grp_path = os.path.join(tmp_path, "test_group_metadata")
    lt.Group._create(ctx, grp_path)
    grp = lt.Group(ctx, grp_path, lt.QueryType.WRITE)

    subgrp_path = os.path.join(tmp_path, "test_group_0")
    lt.Group._create(ctx, subgrp_path)
    grp.add(subgrp_path)
    grp.close()

    grp._open(lt.QueryType.READ)
    assert grp.__len__() == 1
    member = grp._member(0)
    assert os.path.basename(member.uri) == os.path.basename(subgrp_path)
    assert member._type == lt.ObjectType.GROUP
    grp.close()

    grp._open(lt.QueryType.WRITE)
    grp.remove(subgrp_path)
    grp.close()

    grp = lt.Group(ctx, grp_path, lt.QueryType.READ)
    assert grp.__len__() == 0
    grp.close()
