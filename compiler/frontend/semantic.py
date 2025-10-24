"""
Semantic Analysis - Type checking and validation

This module performs semantic analysis on Python AST:
- Type checking and inference
- Variable scope validation
- Control flow analysis
- Detect undefined variables
- Validate operations

Phase: 1.1 (Frontend)
"""

import ast
from typing import Optional, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class TypeKind(Enum):
    """Types supported by the compiler"""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    LIST = "list"
    TUPLE = "tuple"
    DICT = "dict"
    NONE = "None"
    UNKNOWN = "unknown"
    
    def __str__(self):
        return self.value


@dataclass
class Type:
    """Represents a type in the compiler"""
    kind: TypeKind
    element_type: Optional['Type'] = None  # For List[int], etc.
    key_type: Optional['Type'] = None      # For Dict[str, int]
    value_type: Optional['Type'] = None
    
    def __str__(self):
        if self.element_type:
            return f"{self.kind.value}[{self.element_type}]"
        elif self.key_type and self.value_type:
            return f"{self.kind.value}[{self.key_type}, {self.value_type}]"
        return self.kind.value
    
    def is_numeric(self) -> bool:
        return self.kind in (TypeKind.INT, TypeKind.FLOAT)
    
    def is_compatible(self, other: 'Type') -> bool:
        """Check if two types are compatible"""
        if self.kind == TypeKind.UNKNOWN or other.kind == TypeKind.UNKNOWN:
            return True
        if self.kind == other.kind:
            return True
        # Numeric promotion: int -> float
        if self.kind == TypeKind.INT and other.kind == TypeKind.FLOAT:
            return True
        if self.kind == TypeKind.FLOAT and other.kind == TypeKind.INT:
            return True
        return False


@dataclass
class SemanticError:
    """Represents a semantic error"""
    message: str
    line: int
    column: int
    severity: str = "error"  # "error" or "warning"


@dataclass
class SemanticResult:
    """Result of semantic analysis"""
    success: bool
    errors: List[SemanticError] = field(default_factory=list)
    warnings: List[SemanticError] = field(default_factory=list)
    type_map: Dict[ast.AST, Type] = field(default_factory=dict)


class Scope:
    """Represents a lexical scope"""
    
    def __init__(self, parent: Optional['Scope'] = None, name: str = "global"):
        self.parent = parent
        self.name = name
        self.symbols: Dict[str, Type] = {}
        self.children: List['Scope'] = []
        
    def define(self, name: str, typ: Type):
        """Define a variable in this scope"""
        self.symbols[name] = typ
        
    def lookup(self, name: str) -> Optional[Type]:
        """Look up a variable in this scope or parent scopes"""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def lookup_local(self, name: str) -> Optional[Type]:
        """Look up a variable only in this scope"""
        return self.symbols.get(name)


class SemanticAnalyzer(ast.NodeVisitor):
    """
    Performs semantic analysis on Python AST
    
    Checks:
    - Type correctness
    - Variable definitions before use
    - Scope correctness
    - Control flow validity
    """
    
    def __init__(self):
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []
        self.type_map: Dict[ast.AST, Type] = {}
        
        # Scope tracking
        self.global_scope = Scope(name="global")
        self.current_scope = self.global_scope
        
        # Function context
        self.current_function: Optional[ast.FunctionDef] = None
        self.return_type: Optional[Type] = None
        
        # Control flow tracking
        self.in_loop = False
        
    def push_scope(self, name: str = "local"):
        """Enter a new scope"""
        new_scope = Scope(parent=self.current_scope, name=name)
        self.current_scope.children.append(new_scope)
        self.current_scope = new_scope
        
    def pop_scope(self):
        """Exit current scope"""
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent
    
    def error(self, message: str, node: ast.AST):
        """Record a semantic error"""
        self.errors.append(SemanticError(
            message=message,
            line=node.lineno,
            column=node.col_offset,
            severity="error"
        ))
    
    def warning(self, message: str, node: ast.AST):
        """Record a warning"""
        self.warnings.append(SemanticError(
            message=message,
            line=node.lineno,
            column=node.col_offset,
            severity="warning"
        ))
    
    def ast_to_type(self, node: Optional[ast.expr]) -> Type:
        """Convert AST type annotation to internal Type"""
        if node is None:
            return Type(TypeKind.UNKNOWN)
        
        if isinstance(node, ast.Name):
            type_map = {
                'int': TypeKind.INT,
                'float': TypeKind.FLOAT,
                'bool': TypeKind.BOOL,
                'str': TypeKind.STR,
                'None': TypeKind.NONE,
            }
            return Type(type_map.get(node.id, TypeKind.UNKNOWN))
        
        elif isinstance(node, ast.Subscript):
            # Handle List[int], Dict[str, int], etc.
            if isinstance(node.value, ast.Name):
                if node.value.id == 'list' or node.value.id == 'List':
                    elem_type = self.ast_to_type(node.slice)
                    return Type(TypeKind.LIST, element_type=elem_type)
                elif node.value.id == 'dict' or node.value.id == 'Dict':
                    if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                        key_type = self.ast_to_type(node.slice.elts[0])
                        val_type = self.ast_to_type(node.slice.elts[1])
                        return Type(TypeKind.DICT, key_type=key_type, value_type=val_type)
        
        return Type(TypeKind.UNKNOWN)
    
    def infer_type(self, node: ast.expr) -> Type:
        """Infer the type of an expression"""
        # Check if already computed
        if node in self.type_map:
            return self.type_map[node]
        
        typ = Type(TypeKind.UNKNOWN)
        
        if isinstance(node, ast.Constant):
            # Python 3.8+ uses Constant for literals
            if isinstance(node.value, int):
                typ = Type(TypeKind.INT)
            elif isinstance(node.value, float):
                typ = Type(TypeKind.FLOAT)
            elif isinstance(node.value, bool):
                typ = Type(TypeKind.BOOL)
            elif isinstance(node.value, str):
                typ = Type(TypeKind.STR)
            elif node.value is None:
                typ = Type(TypeKind.NONE)
        
        elif isinstance(node, ast.Name):
            # Look up variable type
            var_type = self.current_scope.lookup(node.id)
            if var_type:
                typ = var_type
            else:
                self.error(f"Undefined variable: {node.id}", node)
        
        elif isinstance(node, ast.BinOp):
            # Infer from binary operation
            left_type = self.infer_type(node.left)
            right_type = self.infer_type(node.right)
            
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                if left_type.is_numeric() and right_type.is_numeric():
                    # Float if either operand is float
                    if left_type.kind == TypeKind.FLOAT or right_type.kind == TypeKind.FLOAT:
                        typ = Type(TypeKind.FLOAT)
                    else:
                        typ = Type(TypeKind.INT)
            
            elif isinstance(node.op, ast.FloorDiv):
                typ = Type(TypeKind.INT)
            
            elif isinstance(node.op, ast.Pow):
                typ = Type(TypeKind.FLOAT)
        
        elif isinstance(node, ast.Compare):
            # Comparison always returns bool
            typ = Type(TypeKind.BOOL)
        
        elif isinstance(node, ast.BoolOp):
            # Boolean operations return bool
            typ = Type(TypeKind.BOOL)
        
        elif isinstance(node, ast.UnaryOp):
            operand_type = self.infer_type(node.operand)
            if isinstance(node.op, ast.Not):
                typ = Type(TypeKind.BOOL)
            else:
                typ = operand_type
        
        elif isinstance(node, ast.Call):
            # Function call - need to look up return type
            # For now, mark as unknown
            typ = Type(TypeKind.UNKNOWN)
        
        elif isinstance(node, ast.List):
            # Infer list element type
            if node.elts:
                elem_type = self.infer_type(node.elts[0])
                typ = Type(TypeKind.LIST, element_type=elem_type)
            else:
                typ = Type(TypeKind.LIST, element_type=Type(TypeKind.UNKNOWN))
        
        # Store inferred type
        self.type_map[node] = typ
        return typ
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function definition"""
        old_function = self.current_function
        self.current_function = node
        
        # Get return type
        self.return_type = self.ast_to_type(node.returns)
        
        # Create new scope for function
        self.push_scope(name=node.name)
        
        # Add parameters to scope
        for arg in node.args.args:
            arg_type = self.ast_to_type(arg.annotation)
            self.current_scope.define(arg.arg, arg_type)
            
            if arg_type.kind == TypeKind.UNKNOWN:
                self.warning(f"Parameter '{arg.arg}' has no type hint", node)
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
        
        # Pop function scope
        self.pop_scope()
        
        self.current_function = old_function
        self.return_type = None
    
    def visit_Return(self, node: ast.Return):
        """Check return statement"""
        if node.value is None:
            return_type = Type(TypeKind.NONE)
        else:
            return_type = self.infer_type(node.value)
        
        # Check against declared return type
        if self.return_type and self.return_type.kind != TypeKind.UNKNOWN:
            if not return_type.is_compatible(self.return_type):
                self.error(
                    f"Return type {return_type} incompatible with declared {self.return_type}",
                    node
                )
    
    def visit_Assign(self, node: ast.Assign):
        """Analyze assignment"""
        # Infer value type
        value_type = self.infer_type(node.value)
        
        # Assign to all targets
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Define or update variable
                existing = self.current_scope.lookup_local(target.id)
                if existing and not existing.is_compatible(value_type):
                    self.warning(
                        f"Variable '{target.id}' type changed from {existing} to {value_type}",
                        node
                    )
                self.current_scope.define(target.id, value_type)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Analyze annotated assignment (x: int = 5)"""
        if isinstance(node.target, ast.Name):
            # Get declared type
            declared_type = self.ast_to_type(node.annotation)
            
            # If there's a value, check compatibility
            if node.value:
                value_type = self.infer_type(node.value)
                if not declared_type.is_compatible(value_type):
                    self.error(
                        f"Cannot assign {value_type} to variable of type {declared_type}",
                        node
                    )
            
            # Define variable with declared type
            self.current_scope.define(node.target.id, declared_type)
    
    def visit_AugAssign(self, node: ast.AugAssign):
        """Analyze augmented assignment (+=, -=, etc.)"""
        if isinstance(node.target, ast.Name):
            var_type = self.current_scope.lookup(node.target.id)
            value_type = self.infer_type(node.value)
            
            if var_type and not var_type.is_compatible(value_type):
                self.error(
                    f"Cannot apply {node.op.__class__.__name__} to {var_type} and {value_type}",
                    node
                )
    
    def visit_For(self, node: ast.For):
        """Analyze for loop"""
        old_in_loop = self.in_loop
        self.in_loop = True
        
        # Infer iterator type
        iter_type = self.infer_type(node.iter)
        
        # Define loop variable
        if isinstance(node.target, ast.Name):
            if iter_type.kind == TypeKind.LIST and iter_type.element_type:
                self.current_scope.define(node.target.id, iter_type.element_type)
            else:
                # For range(), assume int
                self.current_scope.define(node.target.id, Type(TypeKind.INT))
        
        # Visit body
        for stmt in node.body:
            self.visit(stmt)
        
        # Visit else clause if present
        for stmt in node.orelse:
            self.visit(stmt)
        
        self.in_loop = old_in_loop
    
    def visit_While(self, node: ast.While):
        """Analyze while loop"""
        old_in_loop = self.in_loop
        self.in_loop = True
        
        # Check condition is boolean-compatible
        cond_type = self.infer_type(node.test)
        
        # Visit body
        for stmt in node.body:
            self.visit(stmt)
        
        self.in_loop = old_in_loop
    
    def visit_If(self, node: ast.If):
        """Analyze if statement"""
        # Check condition
        cond_type = self.infer_type(node.test)
        
        # Visit branches
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
    
    def visit_Break(self, node: ast.Break):
        """Check break statement"""
        if not self.in_loop:
            self.error("'break' outside loop", node)
    
    def visit_Continue(self, node: ast.Continue):
        """Check continue statement"""
        if not self.in_loop:
            self.error("'continue' outside loop", node)


def analyze(tree: ast.Module) -> SemanticResult:
    """
    Perform semantic analysis on AST
    
    Args:
        tree: Python AST module
        
    Returns:
        SemanticResult with errors/warnings and type information
    """
    analyzer = SemanticAnalyzer()
    analyzer.visit(tree)
    
    return SemanticResult(
        success=len(analyzer.errors) == 0,
        errors=analyzer.errors,
        warnings=analyzer.warnings,
        type_map=analyzer.type_map
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("SEMANTIC ANALYZER - Phase 1.1")
    print("=" * 80)
    
    # Test 1: Correct typed function
    print("\n--- Test 1: Well-typed function ---")
    source1 = """
def add(x: int, y: int) -> int:
    result: int = x + y
    return result
"""
    tree1 = ast.parse(source1)
    result1 = analyze(tree1)
    
    if result1.success:
        print("✅ Semantic analysis passed")
    else:
        print("❌ Errors found:")
        for err in result1.errors:
            print(f"  Line {err.line}: {err.message}")
    
    # Test 2: Type mismatch
    print("\n--- Test 2: Type mismatch ---")
    source2 = """
def bad_return(x: int) -> int:
    return "not an int"
"""
    tree2 = ast.parse(source2)
    result2 = analyze(tree2)
    
    if result2.success:
        print("⚠️  No errors detected (should have found type mismatch)")
    else:
        print("✅ Correctly detected errors:")
        for err in result2.errors:
            print(f"  Line {err.line}: {err.message}")
    
    # Test 3: Undefined variable
    print("\n--- Test 3: Undefined variable ---")
    source3 = """
def use_undefined() -> int:
    return x + 1
"""
    tree3 = ast.parse(source3)
    result3 = analyze(tree3)
    
    if result3.success:
        print("⚠️  No errors (should detect undefined)")
    else:
        print("✅ Correctly detected undefined variable:")
        for err in result3.errors:
            print(f"  Line {err.line}: {err.message}")
    
    # Test 4: Break outside loop
    print("\n--- Test 4: Break outside loop ---")
    source4 = """
def bad_break() -> None:
    break
"""
    tree4 = ast.parse(source4)
    result4 = analyze(tree4)
    
    if not result4.success:
        print("✅ Correctly detected break outside loop:")
        for err in result4.errors:
            print(f"  Line {err.line}: {err.message}")
    
    # Test 5: Complex function with loops
    print("\n--- Test 5: Complex function ---")
    source5 = """
def sum_array(arr: list, n: int) -> int:
    total: int = 0
    for i in range(n):
        total += arr[i]
    return total
"""
    tree5 = ast.parse(source5)
    result5 = analyze(tree5)
    
    if result5.success:
        print("✅ Semantic analysis passed")
        print(f"  Warnings: {len(result5.warnings)}")
    else:
        print("❌ Errors found:")
        for err in result5.errors:
            print(f"  Line {err.line}: {err.message}")
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.1 Semantic Analyzer Complete!")
    print("=" * 80)
    print("\nNext: Phase 1.1 - Symbol Table")
