import ast
import numpy as np

import tiledb
from tiledb import _query_condition as qc

"""
A high level wrapper around the Pybind11 query_condition.cc implementation for
filtering query results on attribute values.
"""


class QueryCondition(ast.NodeVisitor):
    """
    Class representing a TileDB query condition object for attribute filtering
    pushdown. Set the query condition with a string representing an expression
    as defined by the grammar below. A more straight forward example of usage is
    given beneath.

    BNF
    ---
    A query condition is made up of one or more Boolean expressions. Multiple
    Boolean expressions are chained together with Boolean operators.

        query_cond ::= bool_expr | bool_expr bool_op query_cond

    A Boolean expression contains a comparison operator. The operator works on a
    TileDB attribute name and value.

        bool_expr ::= attr compare_op val | val compare_op attr

    "and" is the only Boolean operator supported at the moment. We intend to
    support "or" and "not" in future releases.

        bool_op ::= and

    All comparison operators are supported.

        compare_op ::= < | > | <= | >= | == | !=

    TileDB attribute names are strings (no quotes).

        attr ::= <str>

    Values are any Python-valid number or a string enclosed in quotes.

        val ::= <num> | "<str>" | '<str>'

    Example
    -------
    with tiledb.open(uri, mode="r") as A:
        # select cells where the attribute values for foo are less than 5
        # and bar equal to string asdf.
        qc = QueryCondition("foo > 5 and 'asdf' == bar")
        A.query(attrs_filter=qc)
    """

    def __init__(self, expression="", ctx=None):
        if ctx is None:
            ctx = tiledb.default_ctx()

        self._ctx = ctx
        self._schema = None
        self._query_attrs = None
        self._c_obj = None

        self.tree = ast.parse(expression)
        self.raw_str = expression

    def init_query_condition(self, schema, query_attrs):
        self._schema = schema
        self._query_attrs = query_attrs

        if self.tree.body:
            self._c_obj = self.visit(self.tree.body[0])

            if not isinstance(self._c_obj, tiledb._query_condition.qc):
                raise tiledb.TileDBError(
                    "Malformed query condition statement. A query condition must "
                    "be made up of one or more Boolean expressions."
                )

    def visit_Compare(self, node):
        AST_TO_TILEDB = {
            ast.Gt: qc.TILEDB_GT,
            ast.GtE: qc.TILEDB_GE,
            ast.Lt: qc.TILEDB_LT,
            ast.LtE: qc.TILEDB_LE,
            ast.Eq: qc.TILEDB_EQ,
            ast.NotEq: qc.TILEDB_NE,
        }

        try:
            op = AST_TO_TILEDB[type(node.ops[0])]
        except KeyError:
            raise tiledb.TileDBError("Unsupported comparison operator.")

        att = self.visit(node.left)
        val = self.visit(node.comparators[0])

        if not isinstance(att, ast.Name):
            REVERSE_OP = {
                qc.TILEDB_GT: qc.TILEDB_LT,
                qc.TILEDB_GE: qc.TILEDB_LE,
                qc.TILEDB_LT: qc.TILEDB_GT,
                qc.TILEDB_LE: qc.TILEDB_GE,
                qc.TILEDB_EQ: qc.TILEDB_EQ,
                qc.TILEDB_NE: qc.TILEDB_NE,
            }

            op = REVERSE_OP[op]
            att, val = val, att

        if isinstance(att, ast.Name):
            att = att.id
        else:
            raise tiledb.TileDBError("Incorrect type for attribute name.")

        if isinstance(val, ast.Constant):
            sign = val.sign if hasattr(val, "sign") else 1
            val = val.value * sign
        elif isinstance(val, ast.Num):
            # deprecated in 3.8
            sign = val.sign if hasattr(val, "sign") else 1
            val = val.n * sign
        elif isinstance(val, ast.Str) or isinstance(val, ast.Bytes):
            # deprecated in 3.8
            val = val.s
        else:
            raise tiledb.TileDBError(
                f"Incorrect type for comparison value: {ast.dump(val)}"
            )

        if not self._schema.has_attr(att):
            raise tiledb.TileDBError(f"Attribute `{att}` found not in schema.")

        if att not in self._query_attrs:
            raise tiledb.TileDBError(
                f"Attribute `{att}` given to filter in query's `attr_cond` "
                "arg but not found in `attr` arg."
            )

        dtype = self._schema.attr(att).dtype

        if dtype.kind in "SUa":
            dtype_name = "string"
        else:
            try:
                cast = getattr(np, dtype.name)
                val = cast(val)
                dtype_name = dtype.name
            except ValueError:
                raise tiledb.TileDBError(
                    f"Type mismatch between attribute `{att}` and value `{val}`."
                )

        c_obj = qc.qc(self._ctx)
        init_qc = getattr(c_obj, f"init_{dtype_name}")
        init_qc(att, val, op)

        return c_obj

    def visit_BoolOp(self, node):
        AST_TO_TILEDB = {ast.And: qc.TILEDB_AND}

        try:
            op = AST_TO_TILEDB[type(node.op)]
        except KeyError:
            raise tiledb.TileDBError(
                'Unsupported Boolean operator. Only "and" is currently supported.'
            )

        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = result.combine(self.visit(value), op)

        return result

    def visit_Name(self, node):
        return node

    def visit_Constant(self, node):
        return node

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.UAdd):
            sign = 1
        elif isinstance(node.op, ast.USub):
            sign = -1
        else:
            raise tiledb.TileDBError(f"Unsupported UnaryOp type. Saw {ast.dump(node)}.")

        if hasattr(node, "sign"):
            node.operand.sign = node.sign * sign
        else:
            node.operand.sign = sign

        if isinstance(node.operand, ast.UnaryOp):
            return self.visit_UnaryOp(node.operand)
        else:
            return node.operand

    def visit_Num(self, node):
        # deprecated in 3.8
        return node

    def visit_Str(self, node):
        # deprecated in 3.8
        return node

    def visit_Bytes(self, node):
        # deprecated in 3.8
        return node

    def visit_Expr(self, node):
        return self.visit(node.value)

    def __repr__(self) -> str:
        return f'QueryCondition("{self.raw_str}")'
