import ast
import operator as op

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg
}

def safe_eval(expression:str)->float:
    """Evaluate a numeric arithmetic expression safely via AST parsing.
    Supports + - * / // % ** and parentheses. Rejects anything else."""
    def _eval(node):
        if isinstance(node, ast.constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expression, mode='eval')
    return _eval(tree.body)

def calculator(expression:str)->str:
    """Evaluate an arithmetic expression and return the string result or an error message"""
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}':{e}"


## Callable registry of tools
TOOLS = {
    "calculator": calculator,
}    