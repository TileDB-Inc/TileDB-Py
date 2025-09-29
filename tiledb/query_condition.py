import ast
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Type, Union

import numpy as np

import tiledb.main as qc

from .array import Array
from .ctx import Ctx, default_ctx
from .libtiledb import TileDBError

"""
A high level wrapper around the Pybind11 query_condition.cc implementation for
filtering query results on attribute and dimension values.
"""

QueryConditionNodeElem = Union[ast.Name, ast.Constant, ast.Call]


@dataclass
class QueryCondition:
    """
    Class representing a TileDB query condition object for attribute and dimension
    (sparse arrays only) filtering pushdown.

    A query condition is set with a string representing an expression
    as defined by the grammar below. A more straight forward example of usage is
    given beneath.

    When querying a sparse array, only the values that satisfy the given
    condition are returned (coupled with their associated coordinates). An example
    may be found in `examples/query_condition_sparse.py`.

    For dense arrays, the given shape of the query matches the shape of the output
    array. Values that DO NOT satisfy the given condition are filled with the
    TileDB default fill value. Different attribute and dimension types have different
    default fill values as outlined here
    (https://docs.tiledb.com/main/background/internal-mechanics/writing#default-fill-values).
    An example may be found in `examples/query_condition_dense.py`.

    A query condition is made up of one or more Boolean expressions. Multiple
    Boolean expressions are chained together with Boolean operators. The ``or_op``
    Boolean operators are given lower presedence than ``and_op``.

    A Bitwise expression may either be a comparison expression or membership
    expression.

    A Boolean expression may either be a comparison expression or membership
    expression.

    A comparison expression contains a comparison operator. The operator works on a
    TileDB attribute or dimension name (hereby known as a "TileDB variable") and value.

    All comparison operators are supported.

    Bitwise operators are given higher precedence than comparison operators.
    Boolean operators are given lower precedence than comparison operators.

    If an attribute name has special characters in it, you can wrap ``namehere``
    in ``attr("namehere")``.

    A membership expression contains the membership operator, ``in``. The operator
    works on a TileDB variable and list of values.

    TileDB variable names are Python valid variables or a ``attr()`` or ``dim()`` casted string.

    Values are any Python-valid number or string. datetime64 values should first be
    cast to UNIX seconds. Values may also be casted with ``val()``.

    **Example:**

    >>> with tiledb.open(uri, mode="r") as A:
    >>>     # Select cells where the values for `foo` are less than 5
    >>>     # and `bar` equal to string "asdf".
    >>>     # Note precedence is equivalent to:
    >>>     # tiledb.QueryCondition("foo > 5 or ('asdf' == var('b a r') and baz <= val(1.0))")
    >>>     A.query(cond=tiledb.QueryCondition("foo > 5 or 'asdf' == var('b a r') and baz <= val(1.0)"))
    >>>
    >>>     # Select cells where the values for `foo` are equal to 1, 2, or 3.
    >>>     # Note this is equivalent to:
    >>>     # tiledb.QueryCondition("foo == 1 or foo == 2 or foo == 3")
    >>>     A.query(cond=tiledb.QueryCondition("foo in [1, 2, 3]"))
    >>>
    >>>     # Example showing that bitwise operators (| ^ &) are given higher precedence than comparison operators
    >>>     # and comparison operators are given higher precedence than logical operators.
    >>>     # Note this is equivalent to:
    >>>     # tiledb.QueryCondition("((foo == 1) or (foo == 2)) and ('xyz' == var('b a r')) and ((foo & 1) == 0"))
    >>>     A.query(cond=tiledb.QueryCondition("foo == 1 or foo == 2 and 'xyz' == var('b a r') and foo & 1 == 0"))
    """

    expression: str
    ctx: Ctx = field(default_factory=default_ctx, repr=False)
    tree: ast.Expression = field(init=False, repr=False)
    c_obj: qc.PyQueryCondition = field(init=False, repr=False)

    def __post_init__(self):
        try:
            self.tree = ast.parse(f"({self.expression})", mode="eval")
        except:
            raise TileDBError(
                "Could not parse the given QueryCondition statement: "
                f"{self.expression}"
            )

        if not self.tree:
            raise TileDBError(
                "The query condition statement could not be parsed properly. "
                "(Is this an empty expression?)"
            )

    def init_query_condition(self, uri: str, query_attrs: List[str], ctx):
        qctree = QueryConditionTree(
            self.ctx, Array.load_typed(uri, ctx=ctx), query_attrs
        )
        self.c_obj = qctree.visit(self.tree.body)

        if not isinstance(self.c_obj, qc.PyQueryCondition):
            raise TileDBError(
                "Malformed query condition statement. A query condition must "
                "be made up of one or more boolean expressions."
            )


@dataclass
class QueryConditionTree(ast.NodeVisitor):
    ctx: Ctx
    array: Array
    query_attrs: List[str]

    def visit_BitOr(self, node):
        return qc.TILEDB_OR

    def visit_Or(self, node):
        return qc.TILEDB_OR

    def visit_BitAnd(self, node):
        return qc.TILEDB_AND

    def visit_And(self, node):
        return qc.TILEDB_AND

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

    def visit_In(self, node):
        return node

    def visit_NotIn(self, node):
        return node

    def visit_Is(self, node):
        raise TileDBError("the `is` operator is not supported")

    def visit_IsNot(self, node):
        raise TileDBError("the `is not` operator is not supported")

    def visit_List(self, node):
        return list(node.elts)

    def visit_Attribute(self, node) -> qc.PyQueryCondition:
        raise TileDBError(
            f"Unhandled dot operator in {ast.dump(node)} -- if your attribute name "
            'has a dot in it, e.g. `orig.ident`, please wrap it with `attr("...")`, '
            'e.g. `attr("orig.ident")`'
        )

    def visit_Compare(self, node: Type[ast.Compare]) -> qc.PyQueryCondition:
        operator = self.visit(node.ops[0])

        if operator in (
            qc.TILEDB_GT,
            qc.TILEDB_GE,
            qc.TILEDB_LT,
            qc.TILEDB_LE,
            qc.TILEDB_EQ,
            qc.TILEDB_NE,
        ):
            result = self.aux_visit_Compare(
                self.visit(node.left),
                operator,
                self.visit(node.comparators[0]),
            )

            # Handling cases value < variable < value
            for lhs, op, rhs in zip(
                node.comparators[:-1], node.ops[1:], node.comparators[1:]
            ):
                value = self.aux_visit_Compare(
                    self.visit(lhs), self.visit(op), self.visit(rhs)
                )
                result = result.combine(value, qc.TILEDB_AND)
        elif isinstance(operator, (ast.In, ast.NotIn)):
            rhs = node.comparators[0]
            if not isinstance(rhs, ast.List):
                raise TileDBError(
                    "`in` operator syntax must be written as `variable in ['l', 'i', 's', 't']`"
                )

            # For 'my_string in ["red", "yellow"]': node.left is ast.Name
            # For 'attr(my.string) in ["red", "yellow"]': node.left is ast.Call
            if isinstance(node.left, ast.Call):
                if node.left.func.id != "attr":
                    raise TileDBError(
                        f"query condition left-hand side function call must be 'attr'; got '{node.left.func.id}'"
                    )

                if len(node.left.args) != 1:
                    raise TileDBError(
                        f"query condition left-hand side 'attr' function call must have one argument; got '{len(node.left.args)}'"
                    )

                arg = node.left.args[0]
                if not isinstance(arg, ast.Constant):
                    raise TileDBError(
                        "query condition left-hand side 'attr' argument must be a constant"
                    )
                variable = arg.value

            elif isinstance(node.left, ast.Name):
                variable = node.left.id
            else:
                raise TileDBError(
                    f"cannot handle query condition left-hand side of type '{type(node.left)}'"
                )

            values = [self.get_value_from_node(val) for val in self.visit(rhs)]
            if len(values) == 0:
                raise TileDBError(
                    "At least one value must be provided to " "the set membership"
                )

            if self.array.schema.has_attr(variable):
                enum_label = self.array.attr(variable).enum_label
                if enum_label is not None:
                    dt = self.array.enum(enum_label).dtype
                else:
                    dt = self.array.attr(variable).dtype
            else:
                dt = self.array.schema.attr_or_dim_dtype(variable)

            dtype = "string" if dt.kind in "SUa" else dt.name
            op = qc.TILEDB_IN if isinstance(operator, ast.In) else qc.TILEDB_NOT_IN
            result = self.create_pyqc(dtype)(self.ctx, variable, values, op)

        else:
            raise TileDBError(f"unrecognized operator in <<{ast.dump(node)}>>")

        return result

    def aux_visit_Compare(
        self,
        lhs: QueryConditionNodeElem,
        op_node: qc.tiledb_query_condition_op_t,
        rhs: QueryConditionNodeElem,
    ) -> qc.PyQueryCondition:
        variable, value, op = self.order_nodes(lhs, rhs, op_node)

        variable = self.get_variable_from_node(variable)
        value = self.get_value_from_node(value)

        pyqc = qc.PyQueryCondition(self.ctx)

        if value is None:
            self.init_pyqc(pyqc, None)(variable, op)
        else:
            if self.array.schema.has_attr(variable):
                enum_label = self.array.attr(variable).enum_label
                if enum_label is not None:
                    dt = self.array.enum(enum_label).dtype
                else:
                    dt = self.array.attr(variable).dtype
            else:
                dt = self.array.schema.attr_or_dim_dtype(variable)

            dtype = "string" if dt.kind in "SUa" else dt.name
            value = self.cast_value_to_dtype(value, dtype)
            self.init_pyqc(pyqc, dtype)(variable, value, op)

        return pyqc

    def is_variable_node(self, variable: QueryConditionNodeElem) -> bool:
        if isinstance(variable, ast.Call):
            if not isinstance(variable.func, ast.Name):
                raise TileDBError(f"Unrecognized expression {variable.func}.")

            if variable.func.id not in ("attr", "dim", "val"):
                return False

            return (
                isinstance(variable.args[0], ast.Constant)
                or isinstance(variable.args[0], ast.Constant)
                or isinstance(variable.args[0], ast.Constant)
            )

        return isinstance(variable, ast.Name)

    def order_nodes(
        self,
        variable: QueryConditionNodeElem,
        value: QueryConditionNodeElem,
        op: qc.tiledb_query_condition_op_t,
    ) -> Tuple[
        QueryConditionNodeElem,
        QueryConditionNodeElem,
        qc.tiledb_query_condition_op_t,
    ]:
        if not self.is_variable_node(variable):
            REVERSE_OP = {
                qc.TILEDB_GT: qc.TILEDB_LT,
                qc.TILEDB_GE: qc.TILEDB_LE,
                qc.TILEDB_LT: qc.TILEDB_GT,
                qc.TILEDB_LE: qc.TILEDB_GE,
                qc.TILEDB_EQ: qc.TILEDB_EQ,
                qc.TILEDB_NE: qc.TILEDB_NE,
            }

            op = REVERSE_OP[op]
            variable, value = value, variable

        return variable, value, op

    def get_variable_from_node(self, node: QueryConditionNodeElem) -> Any:
        if not self.is_variable_node(node):
            raise TileDBError(f"Incorrect type for variable name: {ast.dump(node)}")

        variable_node = node

        if isinstance(variable_node, ast.Call):
            if not isinstance(variable_node.func, ast.Name):
                raise TileDBError(f"Unrecognized expression {variable_node.func}.")
            variable_node = variable_node.args[0]

        if isinstance(variable_node, ast.Name):
            variable = variable_node.id
        elif isinstance(variable_node, ast.Constant):
            variable = variable_node.value
        else:
            raise TileDBError(
                f"Incorrect type for variable name: {ast.dump(variable_node)}"
            )

        if self.array.schema.domain.has_dim(variable) and not self.array.schema.sparse:
            raise TileDBError(
                "Cannot apply query condition to dimensions on dense arrays. "
                f"{variable} is a dimension."
            )

        if isinstance(node, ast.Call):
            if node.func.id == "attr" and not self.array.schema.has_attr(variable):
                raise TileDBError(f"{node.func.id} is not an attribute.")

            if node.func.id == "dim" and not self.array.schema.domain.has_dim(variable):
                raise TileDBError(f"{node.func.id} is not a dimension.")

        return variable

    def get_value_from_node(self, node: QueryConditionNodeElem) -> Any:
        value_node = node

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TileDBError(f"Unrecognized expression {node.func}.")

            if node.func.id == "val":
                value_node = node.args[0]
            else:
                raise TileDBError(f"Incorrect type for cast value: {node.func.id}")

        if isinstance(value_node, ast.Constant):
            value = value_node.value
        else:
            raise TileDBError(
                f"Incorrect type for comparison value: {ast.dump(value_node)}: right-hand sides must be constant"
                " expressions, not variables -- did you mean to quote the right-hand side as a string?"
            )

        return value

    def cast_value_to_dtype(
        self, value: Union[str, int, float, bytes], dtype: str
    ) -> Union[str, int, float, bytes]:
        if dtype != "string":
            try:
                # this prevents numeric strings ("1", '123.32') from getting
                # casted to numeric types
                if isinstance(value, str):
                    raise TileDBError(f"Cannot cast `{value}` to {dtype}.")

                if np.issubdtype(dtype, np.datetime64):
                    cast = getattr(np, "int64")
                elif np.issubdtype(dtype, bool):
                    cast = getattr(np, "uint8")
                else:
                    cast = getattr(np, dtype)

                value = cast(value)

            except ValueError:
                raise TileDBError(f"Cannot cast `{value}` to {dtype}.")

        return value

    def init_pyqc(self, pyqc: qc.PyQueryCondition, dtype: str) -> Callable:
        if dtype is None:
            dtype = "null"
        elif dtype != "string":
            if np.issubdtype(dtype, np.datetime64):
                dtype = "int64"
            elif np.issubdtype(dtype, bool):
                dtype = "uint8"

        init_fn_name = f"init_{dtype}"

        if not hasattr(pyqc, init_fn_name):
            raise TileDBError(f"PyQueryCondition.{init_fn_name}() not found.")

        return getattr(pyqc, init_fn_name)

    def create_pyqc(self, dtype: str) -> Callable:
        if dtype != "string":
            if np.issubdtype(dtype, np.datetime64):
                dtype = "int64"
            elif np.issubdtype(dtype, bool):
                dtype = "uint8"

        create_fn_name = f"create_{dtype}"

        if not hasattr(qc.PyQueryCondition, create_fn_name):
            raise TileDBError(f"PyQueryCondition.{create_fn_name}() not found.")

        return getattr(qc.PyQueryCondition, create_fn_name)

    def visit_BinOp(self, node: ast.BinOp) -> qc.PyQueryCondition:
        try:
            op = self.visit(node.op)
        except KeyError:
            raise TileDBError(
                f"Unsupported binary operator: {ast.dump(node.op)}. Only & is currently supported."
            )

        result = self.visit(node.left)
        rhs = node.right[1:] if isinstance(node.right, list) else [node.right]
        for value in rhs:
            visited = self.visit(value)
            if not isinstance(result, qc.PyQueryCondition):
                raise Exception(
                    f"Unable to parse expression component {ast.dump(node)} -- did you mean to quote it as a string?"
                )
            result = result.combine(visited, op)
        return result

    def visit_BoolOp(self, node: ast.BoolOp) -> qc.PyQueryCondition:
        try:
            op = self.visit(node.op)
        except KeyError:
            raise TileDBError(f"Unsupported Boolean operator: {ast.dump(node.op)}.")

        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = result.combine(self.visit(value), op)

        return result

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if not isinstance(node.func, ast.Name):
            raise TileDBError(f"Unrecognized expression {node.func}.")

        if node.func.id not in ("attr", "dim", "val"):
            raise TileDBError("Valid casts are attr(), dim(), or val()).")

        if len(node.args) != 1:
            raise TileDBError(
                f"Exactly one argument must be provided to {node.func.id}()."
            )

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp, sign: int = 1):
        if isinstance(node.op, ast.Not):
            operand = self.visit(node.operand)
            if not isinstance(operand, qc.PyQueryCondition):
                raise TileDBError(
                    f"`not` can only be applied to a query condition, got {type(operand)}"
                )
            return operand.negate()
        if isinstance(node.op, ast.UAdd):
            sign *= 1
        elif isinstance(node.op, ast.USub):
            sign *= -1
        else:
            raise TileDBError(f"Unsupported UnaryOp type. Saw {ast.dump(node)}.")

        if isinstance(node.operand, ast.UnaryOp):
            return self.visit_UnaryOp(node.operand, sign)
        else:
            if isinstance(node.operand, ast.Constant):
                node.operand.value *= sign
            elif isinstance(node.operand, ast.Constant):
                node.operand.n *= sign
            else:
                raise TileDBError(
                    f"Unexpected node type following UnaryOp. Saw {ast.dump(node)}."
                )

            return node.operand
