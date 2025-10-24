"""
Intermediate Representation (IR) - Typed IR Nodes

This module defines a custom typed intermediate representation that bridges
Python AST and LLVM IR. The IR is:
- Typed (all nodes have explicit types)
- SSA-form ready (single static assignment)
- Simple to optimize
- Easy to map to LLVM

Phase: 1.2 (IR)
"""

from typing import Optional, List, Union, Dict
from dataclasses import dataclass, field
from enum import Enum

from compiler.frontend.semantic import Type, TypeKind


class IRNodeKind(Enum):
    """Kind of IR node"""
    # Literals
    CONST_INT = "const_int"
    CONST_FLOAT = "const_float"
    CONST_BOOL = "const_bool"
    CONST_STR = "const_str"
    CONST_NONE = "const_none"
    
    # Variables
    VAR = "var"
    LOAD = "load"
    STORE = "store"
    
    # Binary operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FLOORDIV = "floordiv"
    MOD = "mod"
    POW = "pow"
    
    # Bitwise operations
    AND = "and"
    OR = "or"
    XOR = "xor"
    LSHIFT = "lshift"
    RSHIFT = "rshift"
    
    # Logical operations (for boolean and/or)
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    
    # Comparison
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    
    # Unary operations
    NEG = "neg"
    NOT = "not"
    INVERT = "invert"  # Bitwise not (~)
    
    # Control flow
    LABEL = "label"
    JUMP = "jump"
    BRANCH = "branch"
    RETURN = "return"
    
    # Function calls
    CALL = "call"
    
    # Memory
    ALLOCA = "alloca"
    
    # Conversion
    CAST = "cast"
    
    # Collections (Phase 3)
    LIST_LITERAL = "list_literal"
    LIST_INDEX = "list_index"
    LIST_APPEND = "list_append"
    LIST_LEN = "list_len"
    TUPLE_LITERAL = "tuple_literal"
    DICT_LITERAL = "dict_literal"
    DICT_GET = "dict_get"
    DICT_SET = "dict_set"
    
    # Async/Await (Phase 4)
    ASYNC_FUNCTION = "async_function"
    AWAIT = "await"
    ASYNC_CALL = "async_call"
    YIELD = "yield"
    YIELD_FROM = "yield_from"
    
    # Exception Handling (Phase 4)
    TRY = "try"
    EXCEPT = "except"
    FINALLY = "finally"
    RAISE = "raise"
    
    # Context Managers (Phase 4)
    WITH = "with"
    ENTER = "enter"
    EXIT = "exit"
    
    # OOP (Week 2)
    CLASS = "class"
    NEW_OBJECT = "new_object"
    GET_ATTR = "get_attr"
    SET_ATTR = "set_attr"
    METHOD_CALL = "method_call"


@dataclass
class IRNode:
    """Base class for all IR nodes"""
    kind: IRNodeKind
    typ: Type
    line: int = 0
    
    def __str__(self):
        return f"{self.kind.value}: {self.typ}"


@dataclass
class IRConstInt(IRNode):
    """Integer constant"""
    value: int = 0
    
    def __init__(self, value: int, line: int = 0):
        super().__init__(IRNodeKind.CONST_INT, Type(TypeKind.INT), line)
        self.value = value
    
    def __str__(self):
        return f"{self.value}i"


@dataclass
class IRConstFloat(IRNode):
    """Float constant"""
    value: float = 0.0
    
    def __init__(self, value: float, line: int = 0):
        super().__init__(IRNodeKind.CONST_FLOAT, Type(TypeKind.FLOAT), line)
        self.value = value
    
    def __str__(self):
        return f"{self.value}f"


@dataclass
class IRConstBool(IRNode):
    """Boolean constant"""
    value: bool = False
    
    def __init__(self, value: bool, line: int = 0):
        super().__init__(IRNodeKind.CONST_BOOL, Type(TypeKind.BOOL), line)
        self.value = value
    
    def __str__(self):
        return f"{self.value}b"


@dataclass
class IRConstStr(IRNode):
    """String constant"""
    value: str = ""
    
    def __init__(self, value: str, line: int = 0):
        from ..frontend.semantic import Type, TypeKind
        super().__init__(IRNodeKind.CONST_STR, Type(TypeKind.UNKNOWN), line)  # TODO: Add STRING type
        self.value = value
    
    def __str__(self):
        return f'"{self.value}"'


@dataclass
class IRConstNone(IRNode):
    """None constant"""
    
    def __init__(self, line: int = 0):
        from ..frontend.semantic import Type, TypeKind
        super().__init__(IRNodeKind.CONST_NONE, Type(TypeKind.UNKNOWN), line)  # TODO: Add NONE type
    
    def __str__(self):
        return "None"


@dataclass
class IRVar(IRNode):
    """Variable reference"""
    name: str = ""
    
    def __init__(self, name: str, typ: Type, line: int = 0):
        super().__init__(IRNodeKind.VAR, typ, line)
        self.name = name
    
    def __str__(self):
        return f"%{self.name}"


@dataclass
class IRLoad(IRNode):
    """Load from variable"""
    var: IRVar = None
    
    def __init__(self, var: IRVar, line: int = 0):
        super().__init__(IRNodeKind.LOAD, var.typ, line)
        self.var = var
    
    def __str__(self):
        return f"load {self.var}"


@dataclass
class IRStore(IRNode):
    """Store to variable"""
    var: IRVar = None
    value: IRNode = None
    
    def __init__(self, var: IRVar, value: IRNode, line: int = 0):
        super().__init__(IRNodeKind.STORE, Type(TypeKind.NONE), line)
        self.var = var
        self.value = value
    
    def __str__(self):
        return f"store {self.value} -> {self.var}"


@dataclass
class IRBinOp(IRNode):
    """Binary operation"""
    left: IRNode = None
    right: IRNode = None
    
    def __init__(self, kind: IRNodeKind, left: IRNode, right: IRNode, typ: Type, line: int = 0):
        super().__init__(kind, typ, line)
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"{self.kind.value} {self.left}, {self.right}"


@dataclass
class IRUnaryOp(IRNode):
    """Unary operation"""
    operand: IRNode = None
    
    def __init__(self, kind: IRNodeKind, operand: IRNode, typ: Type, line: int = 0):
        super().__init__(kind, typ, line)
        self.operand = operand
    
    def __str__(self):
        return f"{self.kind.value} {self.operand}"


@dataclass
class IRLabel(IRNode):
    """Label for jumps"""
    name: str = ""
    
    def __init__(self, name: str, line: int = 0):
        super().__init__(IRNodeKind.LABEL, Type(TypeKind.NONE), line)
        self.name = name
    
    def __str__(self):
        return f"{self.name}:"


@dataclass
class IRJump(IRNode):
    """Unconditional jump"""
    target: str = ""
    
    def __init__(self, target: str, line: int = 0):
        super().__init__(IRNodeKind.JUMP, Type(TypeKind.NONE), line)
        self.target = target
    
    def __str__(self):
        return f"jump {self.target}"


@dataclass
class IRBranch(IRNode):
    """Conditional branch"""
    condition: IRNode = None
    true_label: str = ""
    false_label: str = ""
    
    def __init__(self, condition: IRNode, true_label: str, false_label: str, line: int = 0):
        super().__init__(IRNodeKind.BRANCH, Type(TypeKind.NONE), line)
        self.condition = condition
        self.true_label = true_label
        self.false_label = false_label
    
    def __str__(self):
        return f"branch {self.condition} ? {self.true_label} : {self.false_label}"


@dataclass
class IRReturn(IRNode):
    """Return from function"""
    value: Optional[IRNode] = None
    
    def __init__(self, value: Optional[IRNode], line: int = 0):
        typ = value.typ if value else Type(TypeKind.NONE)
        super().__init__(IRNodeKind.RETURN, typ, line)
        self.value = value
    
    def __str__(self):
        if self.value:
            return f"return {self.value}"
        return "return"


@dataclass
class IRCall(IRNode):
    """Function call"""
    func_name: str = ""
    args: List[IRNode] = field(default_factory=list)
    
    def __init__(self, func_name: str, args: List[IRNode], return_type: Type, line: int = 0):
        super().__init__(IRNodeKind.CALL, return_type, line)
        self.func_name = func_name
        self.args = args
    
    def __str__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"call @{self.func_name}({args_str})"


@dataclass
class IRAlloca(IRNode):
    """Allocate stack space"""
    name: str = ""
    
    def __init__(self, name: str, typ: Type, line: int = 0):
        super().__init__(IRNodeKind.ALLOCA, typ, line)
        self.name = name
    
    def __str__(self):
        return f"alloca %{self.name}: {self.typ}"


@dataclass
class IRCast(IRNode):
    """Type cast"""
    value: IRNode = None
    target_type: Type = None
    
    def __init__(self, value: IRNode, target_type: Type, line: int = 0):
        super().__init__(IRNodeKind.CAST, target_type, line)
        self.value = value
        self.target_type = target_type
    
    def __str__(self):
        return f"cast {self.value} to {self.target_type}"


@dataclass
class IRBasicBlock:
    """Basic block - sequence of instructions with single entry/exit"""
    name: str
    instructions: List[IRNode] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    
    def add_instruction(self, instr: IRNode):
        """Add instruction to block"""
        self.instructions.append(instr)
    
    def __str__(self):
        lines = [f"{self.name}:"]
        for instr in self.instructions:
            lines.append(f"  {instr}")
        return "\n".join(lines)


@dataclass
class IRFunction:
    """Function definition in IR"""
    name: str
    param_names: List[str]
    param_types: List[Type]
    return_type: Type
    blocks: List[IRBasicBlock] = field(default_factory=list)
    local_vars: List[tuple] = field(default_factory=list)  # (name, type)
    
    def add_block(self, block: IRBasicBlock):
        """Add basic block to function"""
        self.blocks.append(block)
    
    def get_block(self, name: str) -> Optional[IRBasicBlock]:
        """Get block by name"""
        for block in self.blocks:
            if block.name == name:
                return block
        return None
    
    def add_local_var(self, name: str, typ: Type):
        """Add local variable"""
        self.local_vars.append((name, typ))
    
    def __str__(self):
        params = ", ".join(f"%{n}: {t}" for n, t in zip(self.param_names, self.param_types))
        lines = [f"function @{self.name}({params}) -> {self.return_type} {{"]
        
        if self.local_vars:
            lines.append("  ; Local variables")
            for name, typ in self.local_vars:
                lines.append(f"  %{name}: {typ}")
            lines.append("")
        
        for block in self.blocks:
            for line in str(block).split('\n'):
                lines.append(f"  {line}")
            lines.append("")
        
        lines.append("}")
        return "\n".join(lines)


@dataclass
class IRModule:
    """Top-level IR module"""
    name: str
    functions: List[IRFunction] = field(default_factory=list)
    global_vars: List[tuple] = field(default_factory=list)  # (name, type, init_value)
    
    def add_function(self, func: IRFunction):
        """Add function to module"""
        self.functions.append(func)
    
    def get_function(self, name: str) -> Optional[IRFunction]:
        """Get function by name"""
        for func in self.functions:
            if func.name == name:
                return func
        return None
    
    def add_global_var(self, name: str, typ: Type, init_value=None):
        """Add global variable"""
        self.global_vars.append((name, typ, init_value))
    
    def __str__(self):
        lines = [f"; Module: {self.name}", ""]
        
        if self.global_vars:
            lines.append("; Global variables")
            for name, typ, init in self.global_vars:
                lines.append(f"@{name}: {typ} = {init if init else 'undef'}")
            lines.append("")
        
        for func in self.functions:
            lines.append(str(func))
            lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# Phase 3: Collection IR Nodes
# ============================================================================

@dataclass
class IRListLiteral(IRNode):
    """List literal: [elem1, elem2, ...]"""
    elements: List[IRNode] = field(default_factory=list)
    element_type: Type = field(default_factory=lambda: Type(TypeKind.UNKNOWN))
    
    def __init__(self, elements: List[IRNode], element_type: Type, line: int = 0):
        # List type is represented as UNKNOWN for now (should be LIST in future)
        super().__init__(IRNodeKind.LIST_LITERAL, Type(TypeKind.UNKNOWN), line)
        self.elements = elements
        self.element_type = element_type
    
    def __str__(self):
        elem_strs = [str(e) for e in self.elements]
        return f"[{', '.join(elem_strs)}] : List[{self.element_type}]"


@dataclass
class IRListIndex(IRNode):
    """List indexing: list[index]"""
    list_node: IRNode = None
    index: IRNode = None
    
    def __init__(self, list_node: IRNode, index: IRNode, result_type: Type, line: int = 0):
        super().__init__(IRNodeKind.LIST_INDEX, result_type, line)
        self.list_node = list_node
        self.index = index
    
    def __str__(self):
        return f"{self.list_node}[{self.index}]"


@dataclass
class IRListAppend(IRNode):
    """List append: list.append(value)"""
    list_node: IRNode = None
    value: IRNode = None
    
    def __init__(self, list_node: IRNode, value: IRNode, line: int = 0):
        # append returns None
        super().__init__(IRNodeKind.LIST_APPEND, Type(TypeKind.UNKNOWN), line)
        self.list_node = list_node
        self.value = value
    
    def __str__(self):
        return f"{self.list_node}.append({self.value})"


@dataclass
class IRListLen(IRNode):
    """List length: len(list)"""
    list_node: IRNode = None
    
    def __init__(self, list_node: IRNode, line: int = 0):
        super().__init__(IRNodeKind.LIST_LEN, Type(TypeKind.INT), line)
        self.list_node = list_node
    
    def __str__(self):
        return f"len({self.list_node})"


@dataclass
class IRTupleLiteral(IRNode):
    """Tuple literal: (elem1, elem2, ...)"""
    elements: List[IRNode] = field(default_factory=list)
    
    def __init__(self, elements: List[IRNode], line: int = 0):
        super().__init__(IRNodeKind.TUPLE_LITERAL, Type(TypeKind.UNKNOWN), line)
        self.elements = elements
    
    def __str__(self):
        elem_strs = [str(e) for e in self.elements]
        return f"({', '.join(elem_strs)})"


@dataclass
class IRDictLiteral(IRNode):
    """Dictionary literal: {key1: val1, key2: val2, ...}"""
    keys: List[IRNode] = field(default_factory=list)
    values: List[IRNode] = field(default_factory=list)
    
    def __init__(self, keys: List[IRNode], values: List[IRNode], line: int = 0):
        super().__init__(IRNodeKind.DICT_LITERAL, Type(TypeKind.UNKNOWN), line)
        self.keys = keys
        self.values = values
    
    def __str__(self):
        pairs = [f"{k}: {v}" for k, v in zip(self.keys, self.values)]
        return f"{{{', '.join(pairs)}}}"


# ============================================================================
# Phase 4: Async/Await & Advanced Control Flow
# ============================================================================

@dataclass
class IRAsyncFunction(IRNode):
    """Async function definition: async def func(...) -> ..."""
    name: str = ""
    params: List[IRNode] = field(default_factory=list)
    body: List[IRNode] = field(default_factory=list)
    
    def __init__(self, name: str, params: List[IRNode], body: List[IRNode], return_type: Type, line: int = 0):
        super().__init__(IRNodeKind.ASYNC_FUNCTION, return_type, line)
        self.name = name
        self.params = params
        self.body = body
    
    def add_block(self, block):
        """Add basic block to function body (for compatibility with IRFunction)"""
        self.body.append(block)
    
    def __str__(self):
        """Pretty print async/generator function with body details"""
        params_str = ", ".join(f"%{p.name}: {p.typ}" for p in self.params)
        lines = [f"async def @{self.name}({params_str}) -> {self.typ} {{"]
        
        # Show body blocks
        for item in self.body:
            if isinstance(item, IRBasicBlock):
                for line in str(item).split('\n'):
                    lines.append(f"  {line}")
                lines.append("")
        
        lines.append("}")
        return "\n".join(lines)


@dataclass
class IRAwait(IRNode):
    """Await expression: await coroutine()"""
    coroutine: IRNode = None
    
    def __init__(self, coroutine: IRNode, result_type: Type, line: int = 0):
        super().__init__(IRNodeKind.AWAIT, result_type, line)
        self.coroutine = coroutine
    
    def __str__(self):
        return f"await {self.coroutine}"


@dataclass
class IRYield(IRNode):
    """Yield expression: yield value"""
    value: IRNode = None
    
    def __init__(self, value: IRNode, line: int = 0):
        super().__init__(IRNodeKind.YIELD, Type(TypeKind.UNKNOWN), line)
        self.value = value
    
    def __str__(self):
        return f"yield {self.value}"


@dataclass
class IRYieldFrom(IRNode):
    """Yield from expression: yield from iterator"""
    iterator: IRNode = None
    result: IRNode = None  # Temporary to store yielded values
    
    def __init__(self, iterator: IRNode, result: IRNode, line: int = 0):
        super().__init__(IRNodeKind.YIELD_FROM, Type(TypeKind.UNKNOWN), line)
        self.iterator = iterator
        self.result = result
    
    def __str__(self):
        return f"{self.result} = yield from {self.iterator}"


# ============================================================================
# Phase 4: Exception Handling
# ============================================================================

@dataclass
class IRTry(IRNode):
    """Try block: try: body"""
    body: 'IRBasicBlock' = None  # Try block
    except_blocks: List['IRExcept'] = field(default_factory=list)
    finally_block: Optional['IRBasicBlock'] = None
    
    def __init__(self, body: 'IRBasicBlock', except_blocks: List['IRExcept'], 
                 finally_block: Optional['IRBasicBlock'] = None, line: int = 0):
        super().__init__(IRNodeKind.TRY, Type(TypeKind.UNKNOWN), line)
        self.body = body
        self.except_blocks = except_blocks
        self.finally_block = finally_block
    
    def __str__(self):
        return f"try {{ {self.body.name if self.body else 'empty'} }} except {{ {len(self.except_blocks)} handlers }}"


@dataclass
class IRExcept(IRNode):
    """Except block: except ExceptionType as e: handler"""
    exception_type: Optional[Type] = None
    var_name: Optional[str] = None
    handler: List[IRNode] = field(default_factory=list)
    
    def __init__(self, exception_type: Optional[Type], var_name: Optional[str],
                 handler: List[IRNode], line: int = 0):
        super().__init__(IRNodeKind.EXCEPT, Type(TypeKind.UNKNOWN), line)
        self.exception_type = exception_type
        self.var_name = var_name
        self.handler = handler
    
    def __str__(self):
        exc_str = str(self.exception_type) if self.exception_type else "Exception"
        var_str = f" as {self.var_name}" if self.var_name else ""
        return f"except {exc_str}{var_str}"


@dataclass
class IRFinally(IRNode):
    """Finally block: finally: cleanup"""
    body: List[IRNode] = field(default_factory=list)
    
    def __init__(self, body: List[IRNode], line: int = 0):
        super().__init__(IRNodeKind.FINALLY, Type(TypeKind.UNKNOWN), line)
        self.body = body
    
    def __str__(self):
        return f"finally {{ {len(self.body)} stmts }}"


@dataclass
class IRRaise(IRNode):
    """Raise statement: raise exception"""
    exception: Optional[IRNode] = None
    
    def __init__(self, exception: Optional[IRNode] = None, line: int = 0):
        super().__init__(IRNodeKind.RAISE, Type(TypeKind.UNKNOWN), line)
        self.exception = exception
    
    def __str__(self):
        if self.exception:
            return f"raise {self.exception}"
        return "raise"


# ============================================================================
# Phase 4: Context Managers
# ============================================================================

@dataclass
class IRWith(IRNode):
    """With statement: with context as var: body"""
    context_expr: IRNode = None
    var_name: Optional[str] = None
    body: List[IRNode] = field(default_factory=list)
    
    def __init__(self, context_expr: IRNode, var_name: Optional[str],
                 body: List[IRNode], line: int = 0):
        super().__init__(IRNodeKind.WITH, Type(TypeKind.UNKNOWN), line)
        self.context_expr = context_expr
        self.var_name = var_name
        self.body = body
    
    def __str__(self):
        var_str = f" as {self.var_name}" if self.var_name else ""
        return f"with {self.context_expr}{var_str}"


# ============================================================================
# OOP (Object-Oriented Programming) IR Nodes - Week 2 Day 1
# ============================================================================

@dataclass
class IRClass:
    """Class definition"""
    name: str
    base_classes: List[str] = field(default_factory=list)  # List of parent class names
    attributes: Dict[str, Type] = field(default_factory=dict)  # Class attributes
    methods: List['IRFunction'] = field(default_factory=list)  # Method definitions
    line: int = 0
    
    def add_method(self, method: 'IRFunction'):
        """Add a method to the class"""
        self.methods.append(method)
    
    def get_method(self, name: str) -> Optional['IRFunction']:
        """Get method by name"""
        for method in self.methods:
            if method.name == name:
                return method
        return None
    
    def __str__(self):
        bases_str = f"({', '.join(self.base_classes)})" if self.base_classes else ""
        methods_str = ', '.join([m.name for m in self.methods])
        return f"class {self.name}{bases_str}: methods=[{methods_str}]"


class IRNewObject(IRNode):
    """Create new object instance: obj = ClassName(args)"""
    
    def __init__(self, class_name: str, args: List[IRNode], result: IRVar, line: int = 0):
        super().__init__(IRNodeKind.NEW_OBJECT, result.typ, line)
        self.class_name = class_name
        self.args = args
        self.result = result
    
    def __str__(self):
        args_str = ', '.join([str(arg) for arg in self.args])
        return f"{self.result.name} = new {self.class_name}({args_str})"


class IRGetAttr(IRNode):
    """Get object attribute: value = obj.attribute"""
    attribute: str
    result: IRVar
    
    def __init__(self, object: IRNode, attribute: str, result: IRVar, line: int = 0):
        super().__init__(IRNodeKind.GET_ATTR, result.typ, line)
        self.object = object
        self.attribute = attribute
        self.result = result
    
    def __str__(self):
        return f"{self.result.name} = {self.object}.{self.attribute}"


class IRSetAttr(IRNode):
    """Set object attribute: obj.attribute = value"""
    
    def __init__(self, object: IRNode, attribute: str, value: IRNode, line: int = 0):
        super().__init__(IRNodeKind.SET_ATTR, Type(TypeKind.UNKNOWN), line)
        self.object = object
        self.attribute = attribute
        self.value = value
    
    def __str__(self):
        return f"{self.object}.{self.attribute} = {self.value}"


class IRMethodCall(IRNode):
    """Call object method: result = obj.method(args)"""
    
    def __init__(self, object: IRNode, method: str, args: List[IRNode], 
                 result: IRVar, line: int = 0):
        super().__init__(IRNodeKind.METHOD_CALL, result.typ if result else Type(TypeKind.UNKNOWN), line)
        self.object = object
        self.method = method
        self.args = args
        self.result = result
    
    def __str__(self):
        args_str = ', '.join([str(arg) for arg in self.args])
        result_str = f"{self.result.name} = " if self.result else ""
        return f"{result_str}{self.object}.{self.method}({args_str})"


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("INTERMEDIATE REPRESENTATION (IR) - Phase 1.2")
    print("=" * 80)
    
    # Create a simple function: add(x: int, y: int) -> int { return x + y; }
    print("\n--- Building IR for: add(x: int, y: int) -> int ---")
    
    func = IRFunction(
        name="add",
        param_names=["x", "y"],
        param_types=[Type(TypeKind.INT), Type(TypeKind.INT)],
        return_type=Type(TypeKind.INT)
    )
    
    # Entry block
    entry = IRBasicBlock("entry")
    
    # Load parameters
    x_var = IRVar("x", Type(TypeKind.INT))
    y_var = IRVar("y", Type(TypeKind.INT))
    x_load = IRLoad(x_var)
    y_load = IRLoad(y_var)
    
    # Add operation
    add_result = IRBinOp(IRNodeKind.ADD, x_load, y_load, Type(TypeKind.INT))
    
    # Return
    ret = IRReturn(add_result)
    
    entry.add_instruction(x_load)
    entry.add_instruction(y_load)
    entry.add_instruction(add_result)
    entry.add_instruction(ret)
    
    func.add_block(entry)
    
    print(func)
    
    # Create a more complex function with control flow
    print("\n" + "=" * 80)
    print("--- Building IR for: max(a: int, b: int) -> int ---")
    print("=" * 80)
    
    max_func = IRFunction(
        name="max",
        param_names=["a", "b"],
        param_types=[Type(TypeKind.INT), Type(TypeKind.INT)],
        return_type=Type(TypeKind.INT)
    )
    
    # Entry block
    max_entry = IRBasicBlock("entry")
    a_var = IRVar("a", Type(TypeKind.INT))
    b_var = IRVar("b", Type(TypeKind.INT))
    a_load = IRLoad(a_var)
    b_load = IRLoad(b_var)
    
    # Compare a > b
    cmp = IRBinOp(IRNodeKind.GT, a_load, b_load, Type(TypeKind.BOOL))
    
    # Branch
    branch = IRBranch(cmp, "then", "else")
    
    max_entry.add_instruction(a_load)
    max_entry.add_instruction(b_load)
    max_entry.add_instruction(cmp)
    max_entry.add_instruction(branch)
    
    # Then block (return a)
    then_block = IRBasicBlock("then")
    then_block.add_instruction(IRReturn(a_load))
    
    # Else block (return b)
    else_block = IRBasicBlock("else")
    else_block.add_instruction(IRReturn(b_load))
    
    max_func.add_block(max_entry)
    max_func.add_block(then_block)
    max_func.add_block(else_block)
    
    print(max_func)
    
    # Create a module
    print("\n" + "=" * 80)
    print("--- Complete IR Module ---")
    print("=" * 80)
    
    module = IRModule(name="test_module")
    module.add_function(func)
    module.add_function(max_func)
    
    print(module)
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.2 IR Design Complete!")
    print("=" * 80)
    print("\nNext: Phase 1.2 - AST to IR Lowering")
