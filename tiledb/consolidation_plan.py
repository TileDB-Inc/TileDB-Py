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
        :raises TypeError: All dimensions must have the same dtype
        """
        if ctx is None:
            ctx = default_ctx()

        if not isinstance(array, Array):
            raise ValueError("`array` argument must be of type Array")

        if not isinstance(fragment_size, int):
            raise ValueError("`fragment_size` argument must be of type int")

        super().__init__(ctx, lt.Array(ctx, array), fragment_size)

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
