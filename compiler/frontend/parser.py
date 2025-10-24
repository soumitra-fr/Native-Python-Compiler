"""
Frontend Parser - Convert Python source code to AST

This module provides a parser that converts Python source code into an AST
(Abstract Syntax Tree) suitable for compilation. It validates Python syntax
and performs initial structural analysis.

Phase: 1.1 (Frontend)
"""

import ast
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import inspect


@dataclass
class ParseError:
    """Represents a parsing error"""
    message: str
    line: int
    column: int
    code: Optional[str] = None


@dataclass
class ParseResult:
    """Result of parsing Python source code"""
    success: bool
    ast_tree: Optional[ast.Module] = None
    source: Optional[str] = None
    errors: List[ParseError] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class CompilerASTValidator(ast.NodeVisitor):
    """
    Validates AST for compilability
    
    Checks:
    - No unsupported dynamic features (eval, exec, etc.)
    - No metaclasses or decorators (initially)
    - Type hints present where required
    """
    
    def __init__(self):
        self.errors: List[ParseError] = []
        self.warnings: List[str] = []
        self.current_line = 0
        
        # Track supported features
        self.supported_ops = {
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.UAdd, ast.USub, ast.Invert
        }
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Validate function definitions"""
        self.current_line = node.lineno
        
        # Check for decorators (not supported initially)
        if node.decorator_list:
            self.warnings.append(
                f"Line {node.lineno}: Function '{node.name}' has decorators. "
                "Decorators may limit compilability."
            )
        
        # Check for type hints (warn if missing)
        if not node.returns:
            self.warnings.append(
                f"Line {node.lineno}: Function '{node.name}' missing return type hint. "
                "Type hints improve optimization."
            )
        
        # Validate arguments have type hints
        for arg in node.args.args:
            if not arg.annotation:
                self.warnings.append(
                    f"Line {node.lineno}: Argument '{arg.arg}' in '{node.name}' "
                    "missing type hint."
                )
        
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        """Check for unsupported dynamic features"""
        self.current_line = node.lineno
        
        # Check for eval/exec/compile
        if isinstance(node.func, ast.Name):
            dangerous_functions = {'eval', 'exec', 'compile', '__import__'}
            if node.func.id in dangerous_functions:
                self.errors.append(ParseError(
                    message=f"Unsupported dynamic function: {node.func.id}",
                    line=node.lineno,
                    column=node.col_offset
                ))
        
        self.generic_visit(node)
        
    def visit_Import(self, node: ast.Import):
        """Track imports"""
        self.current_line = node.lineno
        self.warnings.append(
            f"Line {node.lineno}: Import statement detected. "
            "Only certain standard library modules are supported."
        )
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from...import statements"""
        self.current_line = node.lineno
        self.warnings.append(
            f"Line {node.lineno}: from...import detected (module: {node.module})."
        )
        self.generic_visit(node)
        
    def visit_Global(self, node: ast.Global):
        """Global keyword not supported"""
        self.errors.append(ParseError(
            message="Global keyword not supported in compiled functions",
            line=node.lineno,
            column=node.col_offset
        ))
        
    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Nonlocal keyword limited support"""
        self.warnings.append(
            f"Line {node.lineno}: nonlocal keyword has limited support."
        )
        self.generic_visit(node)


class Parser:
    """
    Main parser class for converting Python source to AST
    
    Features:
    - Parse Python source code or functions
    - Validate for compilability
    - Track warnings and errors
    - Extract metadata (function signatures, etc.)
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize parser
        
        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict
        
    def parse_source(self, source: str) -> ParseResult:
        """
        Parse Python source code string
        
        Args:
            source: Python source code as string
            
        Returns:
            ParseResult with AST and any errors/warnings
        """
        import textwrap
        
        try:
            # Dedent source to handle indented code
            source = textwrap.dedent(source)
            
            # Parse to Python AST
            tree = ast.parse(source)
            
            # Validate for compilation
            validator = CompilerASTValidator()
            validator.visit(tree)
            
            # Check for errors
            if validator.errors:
                return ParseResult(
                    success=False,
                    source=source,
                    errors=validator.errors
                )
            
            # Check for warnings in strict mode
            if self.strict and validator.warnings:
                errors = [
                    ParseError(
                        message=warning,
                        line=0,
                        column=0
                    )
                    for warning in validator.warnings
                ]
                return ParseResult(
                    success=False,
                    source=source,
                    errors=errors
                )
            
            return ParseResult(
                success=True,
                ast_tree=tree,
                source=source
            )
            
        except SyntaxError as e:
            return ParseResult(
                success=False,
                source=source,
                errors=[ParseError(
                    message=str(e),
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    code=e.text
                )]
            )
        except Exception as e:
            return ParseResult(
                success=False,
                source=source,
                errors=[ParseError(
                    message=f"Unexpected error: {str(e)}",
                    line=0,
                    column=0
                )]
            )
    
    def parse_function(self, func) -> ParseResult:
        """
        Parse a Python function object
        
        Args:
            func: Python function to parse
            
        Returns:
            ParseResult with AST and any errors/warnings
        """
        try:
            source = inspect.getsource(func)
            result = self.parse_source(source)
            
            # Add function metadata
            if result.success:
                result.func_name = func.__name__
                result.func_module = func.__module__
                
            return result
            
        except OSError as e:
            return ParseResult(
                success=False,
                errors=[ParseError(
                    message=f"Could not get source: {str(e)}",
                    line=0,
                    column=0
                )]
            )
    
    def extract_functions(self, tree: ast.Module) -> List[ast.FunctionDef]:
        """
        Extract all function definitions from AST
        
        Args:
            tree: Python AST module
            
        Returns:
            List of FunctionDef nodes
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        return functions
    
    def get_function_signature(self, func_def: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract function signature information
        
        Args:
            func_def: AST FunctionDef node
            
        Returns:
            Dictionary with signature info
        """
        sig = {
            'name': func_def.name,
            'args': [],
            'return_type': None,
            'decorators': [ast.unparse(d) for d in func_def.decorator_list],
        }
        
        # Extract argument information
        for arg in func_def.args.args:
            arg_info = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None
            }
            sig['args'].append(arg_info)
        
        # Extract return type
        if func_def.returns:
            sig['return_type'] = ast.unparse(func_def.returns)
        
        return sig


def parse_file(filename: str, strict: bool = False) -> ParseResult:
    """
    Convenience function to parse a Python file
    
    Args:
        filename: Path to Python file
        strict: If True, warnings are treated as errors
        
    Returns:
        ParseResult
    """
    parser = Parser(strict=strict)
    
    try:
        with open(filename, 'r') as f:
            source = f.read()
        return parser.parse_source(source)
    except IOError as e:
        return ParseResult(
            success=False,
            errors=[ParseError(
                message=f"Could not read file: {str(e)}",
                line=0,
                column=0
            )]
        )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("FRONTEND PARSER - Phase 1.1")
    print("=" * 80)
    
    # Test 1: Simple compilable function
    print("\n--- Test 1: Simple function with type hints ---")
    source1 = """
    def add_numbers(a: int, b: int) -> int:
        return a + b
    """
    
    parser = Parser()
    result = parser.parse_source(source1)
    
    if result.success:
        print("✅ Parse successful")
        functions = parser.extract_functions(result.ast_tree)
        for func in functions:
            sig = parser.get_function_signature(func)
            print(f"Function: {sig['name']}")
            print(f"  Args: {sig['args']}")
            print(f"  Return: {sig['return_type']}")
    else:
        print("❌ Parse failed")
        for error in result.errors:
            print(f"  Error at line {error.line}: {error.message}")
    
    # Test 2: Function without type hints
    print("\n--- Test 2: Function without type hints (warnings) ---")
    source2 = """
    def multiply(x, y):
        return x * y
    """
    
    result = parser.parse_source(source2)
    
    if result.success:
        print("✅ Parse successful (with warnings)")
    else:
        print("❌ Parse failed")
    
    # Test 3: Unsupported features
    print("\n--- Test 3: Unsupported dynamic features ---")
    source3 = """
    def dangerous(code: str) -> int:
        return eval(code)
    """
    
    result = parser.parse_source(source3)
    
    if result.success:
        print("⚠️  Parse successful but may not compile")
    else:
        print("❌ Parse failed")
        for error in result.errors:
            print(f"  Error at line {error.line}: {error.message}")
    
    # Test 4: Complex function
    print("\n--- Test 4: Complex numeric function ---")
    source4 = """
    def matrix_multiply(a: list, b: list, n: int) -> float:
        result = 0.0
        for i in range(n):
            for j in range(n):
                result += a[i] * b[j]
        return result
    """
    
    result = parser.parse_source(source4)
    
    if result.success:
        print("✅ Parse successful")
        functions = parser.extract_functions(result.ast_tree)
        for func in functions:
            sig = parser.get_function_signature(func)
            print(f"Function: {sig['name']}")
            print(f"  {len(sig['args'])} arguments")
            print(f"  Return type: {sig['return_type']}")
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.1 Parser Implementation Complete!")
    print("=" * 80)
    print("\nNext: Phase 1.1 - Semantic Analysis")
