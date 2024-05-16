import pprint

import tiledb.cc as lt

from .ctx import Ctx, CtxMixin, default_ctx
from .libtiledb import Array


class ConsolidationPlan(CtxMixin, lt.ConsolidationPlan):
    """Represents TileDB ConsolidationPlan class"""

    def __init__(self, ctx: Ctx, array: lt.Array, fragment_size: int):
        """Class representing the consolidation plan for an array. The plan divides the fragments of the array into different nodes that need to be consolidated together.

        :param ctx: A TileDB Context
        :param array: The array
        :param fragment_size: The desired fragment size
        """
        if ctx is None:
            ctx = default_ctx()

        if not isinstance(array, Array):
            raise ValueError("`array` argument must be of type Array")

        if not isinstance(fragment_size, int):
            raise ValueError("`fragment_size` argument must be of type int")

        super().__init__(ctx, lt.Array(ctx, array), fragment_size)

    def __len__(self):
        """Returns the number of nodes in the consolidation plan"""
        return self.num_nodes

    def __repr__(self):
        attrs = {
            "num_nodes": self.num_nodes,
            "fragments": {
                f"node_{node_idx}": {
                    "num_fragments": self.num_fragments(node_idx),
                    "fragment_uris": [
                        self.fragment_uri(node_idx, fragment_idx)
                        for fragment_idx in range(self.num_fragments(node_idx))
                    ],
                }
                for node_idx in range(self.num_nodes)
            },
        }

        return pprint.PrettyPrinter().pformat(attrs)

    def _repr_html_(self):
        from io import StringIO

        output = StringIO()
        output.write("<section>\n")

        output.write("<h3>Consolidation Plan</h3>\n")
        output.write("<table>\n")
        output.write(
            "<tr><th>Node</th><th>Num Fragments</th><th>Fragment URIs</th></tr>\n"
        )
        for node_idx in range(self.num_nodes):
            output.write(
                f"<tr><td>{node_idx}</td><td>{self.num_fragments(node_idx)}</td><td>{', '.join(self.fragment_uri(node_idx, fragment_idx) for fragment_idx in range(self.num_fragments(node_idx)))}</td></tr>\n"
            )
        output.write("</table>\n")

        output.write("</section>\n")
        return output.getvalue()

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_nodes:
            raise IndexError("Index out of bounds")

        return {
            "num_fragments": self.num_fragments(idx),
            "fragment_uris": [
                self.fragment_uri(idx, fragment_idx)
                for fragment_idx in range(self.num_fragments(idx))
            ],
        }

    @property
    def num_nodes(self) -> int:
        """
        :rtype: int
        :return: The number of nodes in the consolidation plan
        """
        return self._num_nodes

    def num_fragments(self, node_idx: int) -> int:
        """
        :param node_idx: Node index to retrieve the data for
        :rtype: int
        :return: The number of fragments for a node in the consolidation plan
        """
        return self._num_fragments(node_idx)

    def fragment_uri(self, node_idx: int, fragment_idx: int) -> str:
        """
        :param node_idx: Node index to retrieve the data for
        :param fragment_idx: Fragment index to retrieve the data for
        :rtype: str
        :return: The fragment URI for a node/fragment in the consolidation plan
        """
        return self._fragment_uri(node_idx, fragment_idx)

    def dump(self) -> str:
        """
        :rtype: str
        :return: The JSON string for the plan
        """
        return self._dump()
