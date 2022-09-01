#%%
from typing import List
from dataclasses import dataclass

from tkinter import W
import tiledb
import tiledb.cc as lt

import numpy as np
import random
import tempfile
import time
import logging

from enum import IntEnum

#%%
def create_array(uri):
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            *[
                tiledb.Dim(
                    name="idx",
                    domain=(-10, 10),
                    tile=2,
                    dtype="float64",
                )
            ]
        ),
        attrs=[tiledb.Attr(name="data", dtype="float64", var=False)],
        cell_order="row-major",
        tile_order="row-major",
        capacity=10000,
        sparse=True,
        allows_duplicates=False,  # TODO parametrize
    )

    logging.debug(f"!! create array at URI: {uri} with schema:\n")
    logging.debug(str(schema))
    tiledb.SparseArray.create(uri, schema)


class OpType(IntEnum):
    DEL = 1
    VAC = 2
    WRT = 3
    CON = 4

    @classmethod
    def from_value(cls, v):
        return OpType(v)


@dataclass
class Step:
    timestamp: int
    op: OpType


@dataclass
class Tape:
    time: List[int]
    ops: List[int]

    @classmethod
    def from_length(cls, length, start=0):
        # create weakly monotonic time
        time0 = np.random.randint(0, 2, length)
        time0[0] = start
        time = np.cumsum(time0)

        # create random sequence of operations
        ops = np.random.randint(1, 5, length)

        return cls(time, ops)

    def __getitem__(self, idx) -> Step:
        return Step(int(self.time[idx]), OpType.from_value(self.ops[idx]))

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class DelModel:
    uri: str

    def __init__(self, uri: str):
        logging.debug(f"initialized DelModel with URI: {uri}")
        self.uri = uri

    def delete(self, timestamp, expr):
        pyschema = tiledb.ArraySchema.load(self.uri)

        pyqc = tiledb.QueryCondition(expr)
        pyqc.init_query_condition(pyschema, ["data"])

        ctx = lt.Context()

        arr = lt.Array(ctx, self.uri, lt.QueryType.DELETE, timestamp)
        q = lt.Query(ctx, arr, lt.QueryType.DELETE)
        q.set_condition(pyqc)

        logging.debug(f"** delete, timestamp={timestamp}, expr='{expr}'")
        assert q.submit() == lt.QueryStatus.COMPLETE

    def consolidate(self, /, purge: bool = False):
        cfg = tiledb.Config()
        if purge:
            cfg["sm.consolidation.purge_deleted_cells"] = "true"

        with tiledb.scope_ctx(cfg):
            logging.debug(f"** consolidate, purge={purge}")
            tiledb.consolidate(self.uri, config=cfg)

            pass

    def vacuum(self):
        logging.debug("** vacuum")
        tiledb.vacuum(self.uri)

    def write(self, timestamp, idx, data):
        logging.debug(f"** write: ts={timestamp}, idx='{idx}', data='{data}'")
        with tiledb.open(self.uri, "w", timestamp=timestamp) as A:
            A[idx] = {"data": data}

    def _eval(self, step: Step):
        op = OpType.from_value(step.op)
        ts = step.timestamp

        if op == OpType.WRT:
            self.write(ts, idx=[2], data=[2])

        elif op == OpType.VAC:
            self.vacuum()

        elif op == OpType.CON:
            purge = bool(random.getrandbits(1))
            self.consolidate(purge=purge)

        elif op == OpType.DEL:
            self.delete(ts, "data == 2")

        else:
            raise ValueError(f"Unsupported op type: '{step.op}'")

    def eval_tape(self, tape: Tape):
        for step in tape:
            self._eval(step)


#%%
logging.getLogger().setLevel(logging.DEBUG)


def run_tape(tape):

    uri = tempfile.mkdtemp()
    create_array(uri)

    m = DelModel(uri)
    m.write(1, np.arange(0, 5), np.arange(0, 5))
    m.write(2, np.arange(5, 10), np.arange(5, 10))

    m.eval_tape(tape)


def exec_one(executor, tape_len=100):
    t = Tape.from_length(100, start=3)

    # try:
    #    pass
    # catch:
    #    pass


#%%
# from concurrent import futures
#
# def doit():
#    with futures.
#%%
# %%
