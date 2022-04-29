import ast
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Callable, List, Tuple, Type, Union

import tiledb
import tiledb.main as qc
from tiledb.main import PyQueryCondition

"""
A high level wrapper around the Pybind11 query_condition.cc implementation for
filtering query results on attribute values.
"""

QueryConditionNodeElem = Union[
    ast.Name, ast.Constant, ast.Call, ast.Num, ast.Str, ast.Bytes
]


@dataclass
class QueryCondition:
    """
    Class representing a TileDB query condition object for attribute filtering
    pushdown. Set the query condition with a string representing an expression
    as defined by the grammar below. A more straight forward example of usage is
    given beneath.

    **BNF:**

    A query condition is made up of one or more Boolean expressions. Multiple
    Boolean expressions are chained together with Boolean operators.

        ``query_cond ::= bool_expr | bool_expr bool_op query_cond``

    A Boolean expression contains a comparison operator. The operator works on a
    TileDB attribute name and value.

        ``bool_expr ::= attr compare_op val | val compare_op attr | val compare_op attr compare_op val``

    ``and`` and ``&`` are the only Boolean operators supported at the moment. We
    intend to support ``or`` and ``not`` in future releases.

        ``bool_op ::= and | &``

    All comparison operators are supported.

        ``compare_op ::= < | > | <= | >= | == | !=``

    TileDB attribute names are Python valid variables or a ``attr()`` casted string.

        ``attr ::= <variable> | attr(<str>)``

    Values are any Python-valid number or string. They may also be casted with ``val()``.

        ``val ::= <num> | <str> | val(val)``

    **Example:**

    >>> with tiledb.open(uri, mode="r") as A:
    >>>     # select cells where the attribute values for foo are less than 5
    >>>     # and bar equal to string asdf.
    >>>     qc = QueryCondition("foo > 5 and 'asdf' == attr('b a r') and baz <= val(1.0)")
    >>>     A.query(attr_cond=qc)
    """

    expression: str
    ctx: tiledb.Ctx = field(default_factory=tiledb.default_ctx, repr=False)
    tree: ast.Expression = field(init=False, repr=False)
    c_obj: PyQueryCondition = field(init=False, repr=False)

    def __post_init__(self):
        try:
            self.tree = ast.parse(self.expression, mode="eval")
        except:
            raise tiledb.TileDBError(
                "Could not parse the given QueryCondition statement: "
                f"{self.expression}"
            )

        if not self.tree:
            raise tiledb.TileDBError(
                "The query condition statement could not be parsed properly. "
                "(Is this an empty expression?)"
            )

    def init_query_condition(self, schema: tiledb.ArraySchema, query_attrs: List[str]):
        qctree = QueryConditionTree(self.ctx, schema, query_attrs)
        self.c_obj = qctree.visit(self.tree.body)

        if not isinstance(self.c_obj, PyQueryCondition):
            raise tiledb.TileDBError(
                "Malformed query condition statement. A query condition must "
                "be made up of one or more Boolean expressions."
            )


@dataclass
class QueryConditionTree(ast.NodeVisitor):
    ctx: tiledb.Ctx
    schema: tiledb.ArraySchema
    query_attrs: List[str]

    def visit_Gt(self, node):
        return qc.TILEDB_GT

    def visit_GtE(self, node):
        return qc.TILEDB_GE

    def visit_Lt(self, node):
        return qc.TILEDB_LT

    def visit_LtE(self, node):
        return qc.TILEDB_LE

    def visit_Eq(self, node):
        return qc.TILEDB_EQ

    def visit_NotEq(self, node):
        return qc.TILEDB_NE

    def visit_BitAnd(self, node):
        return qc.TILEDB_AND

    def visit_And(self, node):
        return qc.TILEDB_AND

    def visit_Compare(self, node: Type[ast.Compare]) -> PyQueryCondition:
        result = self.aux_visit_Compare(
            self.visit(node.left),
            self.visit(node.ops[0]),
            self.visit(node.comparators[0]),
        )

        for lhs, op, rhs in zip(
            node.comparators[:-1], node.ops[1:], node.comparators[1:]
        ):
            value = self.aux_visit_Compare(
                self.visit(lhs), self.visit(op), self.visit(rhs)
            )
            result = result.combine(value, qc.TILEDB_AND)

        return result

    def aux_visit_Compare(
        self,
        lhs: QueryConditionNodeElem,
        op_node: qc.tiledb_query_condition_op_t,
        rhs: QueryConditionNodeElem,
    ) -> PyQueryCondition:
        att, val, op = self.order_nodes(lhs, rhs, op_node)

        att = self.get_att_from_node(att)
        val = self.get_val_from_node(val)

        dt = self.schema.attr(att).dtype
        dtype = "string" if dt.kind in "SUa" else dt.name
        val = self.cast_val_to_dtype(val, dtype)

        pyqc = PyQueryCondition(self.ctx)

        try:
            self.init_pyqc(pyqc, dtype)(att, val, op)
        except tiledb.TileDBError as e:
            raise tiledb.TileDBError(e)

        return pyqc

    def is_att_node(self, att: QueryConditionNodeElem) -> bool:
        if isinstance(att, ast.Call):
            if not isinstance(att.func, ast.Name):
                raise tiledb.TileDBError(f"Unrecognized expression {att.func}.")

            if att.func.id != "attr":
                return False

            return (
                isinstance(att.args[0], ast.Constant)
                or isinstance(att.args[0], ast.Str)
                or isinstance(att.args[0], ast.Bytes)
            )

        return isinstance(att, ast.Name)

    def order_nodes(
        self,
        att: QueryConditionNodeElem,
        val: QueryConditionNodeElem,
        op: qc.tiledb_query_condition_op_t,
    ) -> Tuple[
        QueryConditionNodeElem,
        QueryConditionNodeElem,
        qc.tiledb_query_condition_op_t,
    ]:
        if not self.is_att_node(att):
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

        return att, val, op

    def get_att_from_node(self, node: QueryConditionNodeElem) -> Any:
        if self.is_att_node(node):
            att_node = node

            if isinstance(att_node, ast.Call):
                if not isinstance(att_node.func, ast.Name):
                    raise tiledb.TileDBError(
                        f"Unrecognized expression {att_node.func}."
                    )
                att_node = att_node.args[0]

            if isinstance(att_node, ast.Name):
                att = att_node.id
            elif isinstance(att_node, ast.Constant):
                att = att_node.value
            elif isinstance(att_node, ast.Str) or isinstance(att_node, ast.Bytes):
                # deprecated in 3.8
                att = att_node.s
            else:
                raise tiledb.TileDBError(
                    f"Incorrect type for attribute name: {ast.dump(att_node)}"
                )
        else:
            raise tiledb.TileDBError(
                f"Incorrect type for attribute name: {ast.dump(node)}"
            )

        if not self.schema.has_attr(att):
            if self.schema.domain.has_dim(att):
                raise tiledb.TileDBError(
                    f"`{att}` is a dimension. QueryConditions currently only "
                    "work on attributes."
                )
            raise tiledb.TileDBError(f"Attribute `{att}` not found in schema.")

        if att not in self.query_attrs:
            raise tiledb.TileDBError(
                f"Attribute `{att}` given to filter in query's `attr_cond` "
                "arg but not found in `attr` arg."
            )

        return att

    def get_val_from_node(self, node: QueryConditionNodeElem) -> Any:
        val_node = node

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise tiledb.TileDBError(f"Unrecognized expression {node.func}.")

            if node.func.id == "val":
                val_node = node.args[0]
            else:
                raise tiledb.TileDBError(
                    f"Incorrect type for cast value: {node.func.id}"
                )

        if isinstance(val_node, ast.Constant):
            val = val_node.value
        elif isinstance(val_node, ast.Num):
            # deprecated in 3.8
            val = val_node.n
        elif isinstance(val_node, ast.Str) or isinstance(val_node, ast.Bytes):
            # deprecated in 3.8
            val = val_node.s
        else:
            raise tiledb.TileDBError(
                f"Incorrect type for comparison value: {ast.dump(val_node)}"
            )

        return val

    def cast_val_to_dtype(
        self, val: Union[str, int, float, bytes], dtype: str
    ) -> Union[str, int, float, bytes]:
        if dtype != "string":
            try:
                # this prevents numeric strings ("1", '123.32') from getting
                # casted to numeric types
                if isinstance(val, str):
                    raise tiledb.TileDBError(f"Cannot cast `{val}` to {dtype}.")
                cast = getattr(np, dtype)
                val = cast(val)
            except ValueError:
                raise tiledb.TileDBError(f"Cannot cast `{val}` to {dtype}.")

        return val

    def init_pyqc(self, pyqc: PyQueryCondition, dtype: str) -> Callable:
        init_fn_name = f"init_{dtype}"

        if not hasattr(pyqc, init_fn_name):
            raise tiledb.TileDBError(f"PyQueryCondition.{init_fn_name}() not found.")

        return getattr(pyqc, init_fn_name)

    def visit_BinOp(self, node: ast.BinOp) -> PyQueryCondition:
        try:
            op = self.visit(node.op)
        except KeyError:
            raise tiledb.TileDBError(
                f"Unsupported binary operator: {ast.dump(node.op)}. Only & is currently supported."
            )

        result = self.visit(node.left)
        rhs = node.right[1:] if isinstance(node.right, list) else [node.right]
        for value in rhs:
            result = result.combine(self.visit(value), op)

        return result

    def visit_BoolOp(self, node: ast.BoolOp) -> PyQueryCondition:
        try:
            op = self.visit(node.op)
        except KeyError:
            raise tiledb.TileDBError(
                f"Unsupported Boolean operator: {ast.dump(node.op)}. "
                'Only "and" is currently supported.'
            )

        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = result.combine(self.visit(value), op)

        return result

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if not isinstance(node.func, ast.Name):
            raise tiledb.TileDBError(f"Unrecognized expression {node.func}.")

        if node.func.id not in ["attr", "val"]:
            raise tiledb.TileDBError(f"Valid casts are attr() or val().")

        if len(node.args) != 1:
            raise tiledb.TileDBError(
                f"Exactly one argument must be provided to {node.func.id}()."
            )

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp, sign: int = 1):
        if isinstance(node.op, ast.UAdd):
            sign *= 1
        elif isinstance(node.op, ast.USub):
            sign *= -1
        else:
            raise tiledb.TileDBError(f"Unsupported UnaryOp type. Saw {ast.dump(node)}.")

        if isinstance(node.operand, ast.UnaryOp):
            return self.visit_UnaryOp(node.operand, sign)
        else:
            if isinstance(node.operand, ast.Constant):
                node.operand.value *= sign
            elif isinstance(node.operand, ast.Num):
                node.operand.n *= sign
            else:
                raise tiledb.TileDBError(
                    f"Unexpected node type following UnaryOp. Saw {ast.dump(node)}."
                )

            return node.operand

    def visit_Num(self, node: ast.Num) -> ast.Num:
        # deprecated in 3.8
        return node

    def visit_Str(self, node: ast.Str) -> ast.Str:
        # deprecated in 3.8
        return node

    def visit_Bytes(self, node: ast.Bytes) -> ast.Bytes:
        # deprecated in 3.8
        return node
