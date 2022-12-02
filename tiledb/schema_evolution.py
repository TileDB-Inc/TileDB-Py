from typing import Optional

import tiledb
from .main import ArraySchemaEvolution as ASE


class ArraySchemaEvolution:
    """This class provides the capability to evolve the ArraySchema of
    a TileDB array in place by adding and removing attributes.
    """

    def __init__(self, ctx: Optional[tiledb.Ctx] = None):
        ctx = ctx or tiledb.default_ctx()

        self.ase = ASE(ctx)

    def add_attribute(self, attr: tiledb.Attr):
        """Add the given attribute to the schema evolution plan.
        Note: this function does not apply any changes; the changes are
        only applied when `ArraySchemaEvolution.array_evolve` is called."""

        self.ase.add_attribute(attr)

    def drop_attribute(self, attr_name: str):
        """Drop the given attribute (by name) in the schema evolution.
        Note: this function does not apply any changes; the changes are
        only applied when `ArraySchemaEvolution.array_evolve` is called."""

        self.ase.drop_attribute(attr_name)

    def array_evolve(self, uri: str):
        """Apply ArraySchemaEvolution actions to Array at given URI."""

        self.ase.array_evolve(uri)

    def timestamp(self, timestamp: int):
        """Sets the timestamp of the schema file."""
        if not isinstance(timestamp, int):
            raise ValueError("'timestamp' argument expects int")

        self.ase.set_timestamp_range(timestamp)
