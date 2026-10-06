"""Safe arithmetic expression parser/evaluator for panel expressions.

Expressions reference other panels by their designation (an array-valued
curve) and free parameters from the variable table (a scalar). Only a small
whitelist of operators and math functions is permitted; the AST is validated
before it ever reaches Python's eval(), and eval() itself is run with no
builtins, so a malformed or malicious expression cannot execute arbitrary
code or take down the app -- it raises a clear, catchable ExpressionError
instead of crashing (this was a common "Invalid expression" crash source in
the original).
"""
from __future__ import annotations

import ast
import re
from typing import Dict, Set

import numpy as np

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ALLOWED_FUNCS = {
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "abs": np.abs,
    "power": np.power, "min": np.minimum, "max": np.maximum,
}

_ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


class ExpressionError(Exception):
    """A user-facing, non-fatal error in an expression."""


class SelfReferenceError(ExpressionError):
    """Raised when an expression (directly or indirectly) references itself."""


def is_valid_designation(name: str) -> bool:
    return bool(IDENTIFIER_RE.match(name)) and name not in _ALLOWED_FUNCS


def parse_expression(expr: str) -> ast.Expression:
    expr = (expr or "").strip()
    if not expr:
        raise ExpressionError("Expression is empty.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_OPS):
                raise ExpressionError("Unsupported operator in expression.")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ExpressionError("Unsupported unary operator in expression.")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                fname = getattr(node.func, "id", None) or getattr(node.func, "attr", "?")
                raise ExpressionError(f"Unknown function '{fname}' in expression.")
            if node.keywords:
                raise ExpressionError("Keyword arguments are not allowed.")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ExpressionError("Only numeric literals are allowed.")
        elif isinstance(node, (ast.Name, ast.Expression, ast.Load, ast.BinOp,
                                ast.UnaryOp, ast.Call) + _ALLOWED_OPS):
            # ast.walk() also visits operator sub-nodes (ast.Add, ast.Mult, ...)
            # as standalone nodes; they were already validated via node.op
            # above, so just let them pass through here.
            pass
        else:
            raise ExpressionError(
                f"Unsupported syntax in expression: {type(node).__name__}."
            )
    return tree


def referenced_names(tree: ast.Expression) -> Set[str]:
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS
    }


def evaluate(tree: ast.Expression, env: Dict[str, object]):
    missing = referenced_names(tree) - set(env)
    if missing:
        raise ExpressionError(
            "Unknown name(s) in expression: " + ", ".join(sorted(missing))
        )
    code = compile(tree, "<expression>", "eval")
    safe_globals = {"__builtins__": {}}
    safe_globals.update(_ALLOWED_FUNCS)
    try:
        with np.errstate(all="ignore"):
            result = eval(code, safe_globals, dict(env))  # noqa: S307 - AST-validated, no builtins
    except ZeroDivisionError as exc:
        raise ExpressionError("Division by zero in expression.") from exc
    except ExpressionError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, never swallowed silently
        raise ExpressionError(f"Error evaluating expression: {exc}") from exc
    return result
