import json
import xml

import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class ConsolidationPlanTest(DiskTestCase):
    def test_consolidation_plan(self):
        path = self.path("test_consolidation_plan")

        array = np.random.rand(4)
        tiledb.from_numpy(path, array)

        with tiledb.open(path, "r") as A:
            cons_plan = tiledb.ConsolidationPlan(tiledb.default_ctx(), A, 2)
            assert cons_plan.num_nodes == 1
            assert cons_plan.num_nodes == len(cons_plan)
            assert cons_plan.num_fragments(0) == 1
            # check that it has a nodes key
            assert "nodes" in json.loads(cons_plan.dump())
            # check that it has a list of nodes
            assert isinstance(json.loads(cons_plan.dump())["nodes"], list)
            # check that each node has a uri key
            for node in json.loads(cons_plan.dump())["nodes"]:
                assert "uri" in node["uris"][0]
            # test __repr__
            try:
                assert (
                    xml.etree.ElementTree.fromstring(cons_plan._repr_html_())
                    is not None
                )
            except:
                pytest.fail(
                    f"Could not parse cons_plan._repr_html_(). Saw {cons_plan._repr_html_()}"
                )
            # test __getitem__
            assert cons_plan[0] == {
                "num_fragments": 1,
                "fragment_uris": [cons_plan.fragment_uri(0, 0)],
            }

        # write a second fragment to the array and check the new consolidation plan
        with tiledb.open(path, "w") as A:
            A[:] = np.random.rand(4)

        with tiledb.open(path, "r") as A:
            cons_plan = tiledb.ConsolidationPlan(tiledb.default_ctx(), A, 4)
            assert cons_plan.num_nodes == 1
            assert cons_plan.num_fragments(0) == 2
