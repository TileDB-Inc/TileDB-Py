from enum import Enum

import tiledb.libtiledb as lt


class DataOrder(Enum):
    increasing = lt.DataOrder.INCREASING_DATA
    decreasing = lt.DataOrder.DECREASING_DATA
    unordered = lt.DataOrder.UNORDERED_DATA
