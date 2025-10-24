"""
Test suite for closures, nested functions, and decorators (Week 1 Day 3)

Tests cover:
1. Nested function definitions
2. Closures (accessing outer scope variables)
3. Basic decorator syntax
4. Multiple decorators
5. Decorator with arguments
"""

import unittest
import sys
import os
import ast

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from compiler.frontend.symbols import SymbolTable
from compiler.ir.lowering import IRLowering
from compiler.backend.llvm_gen import LLVMCodeGen


class TestClosuresDecorators(unittest.TestCase):
    """Test closures, nested functions, and decorators"""
    
    def compile_source(self, source):
        """Helper to compile source code"""
        tree = ast.parse(source)
        symbol_table = SymbolTable(name="global")
        
        lowering = IRLowering(symbol_table)
        ir_module = lowering.visit_Module(tree)
        
        codegen = LLVMCodeGen()
        llvm_module = codegen.generate_module(ir_module)
        
        return ir_module, llvm_module
    
    def test_simple_nested_function(self):
        """Test basic nested function definition"""
        source = """
def outer(x: int) -> int:
    def inner(y: int) -> int:
        return y + 1
    
    result: int = inner(x)
    return result
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Should have outer function
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Simple nested function compiled")
            
        except Exception as e:
            self.fail(f"Simple nested function failed: {e}")
    
    def test_closure_read_only(self):
        """Test closure that reads outer variable"""
        source = """
def make_adder(n: int):
    def add(x: int) -> int:
        return x + n  # Captures n from outer scope
    return add
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Read-only closure compiled")
            
        except Exception as e:
            self.fail(f"Read-only closure failed: {e}")
    
    def test_closure_multiple_variables(self):
        """Test closure capturing multiple outer variables"""
        source = """
def make_multiplier(a: int, b: int):
    def multiply(x: int) -> int:
        return x * a * b  # Captures both a and b
    return multiply
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Multi-variable closure compiled")
            
        except Exception as e:
            self.fail(f"Multi-variable closure failed: {e}")
    
    def test_nested_closure(self):
        """Test nested closures (closure within closure)"""
        source = """
def outer(x: int):
    def middle(y: int):
        def inner(z: int) -> int:
            return x + y + z  # Captures from both outer scopes
        return inner
    return middle
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Nested closure compiled")
            
        except Exception as e:
            self.fail(f"Nested closure failed: {e}")
    
    def test_simple_decorator(self):
        """Test basic decorator syntax"""
        source = """
def my_decorator(func):
    def wrapper():
        return func()
    return wrapper

@my_decorator
def hello() -> int:
    return 42
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Should have decorator and decorated function
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Simple decorator compiled")
            
        except Exception as e:
            self.fail(f"Simple decorator failed: {e}")
    
    def test_decorator_with_function_arg(self):
        """Test decorator that wraps function with arguments"""
        source = """
def trace_decorator(func):
    def wrapper(x: int) -> int:
        result: int = func(x)
        return result
    return wrapper

@trace_decorator
def double(n: int) -> int:
    return n * 2
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Decorator with function arg compiled")
            
        except Exception as e:
            self.fail(f"Decorator with function arg failed: {e}")
    
    def test_multiple_decorators(self):
        """Test stacking multiple decorators"""
        source = """
def decorator1(func):
    def wrapper1() -> int:
        return func()
    return wrapper1

def decorator2(func):
    def wrapper2() -> int:
        return func()
    return wrapper2

@decorator1
@decorator2
def greet() -> int:
    return 1
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Multiple decorators compiled")
            
        except Exception as e:
            self.fail(f"Multiple decorators failed: {e}")
    
    def test_closure_in_decorator(self):
        """Test decorator that creates closure"""
        source = """
def repeat(times: int):
    def decorator(func):
        def wrapper(x: int) -> int:
            result: int = 0
            i: int = 0
            while i < times:
                result = func(x)
                i = i + 1
            return result
        return wrapper
    return decorator

@repeat(3)
def process(n: int) -> int:
    return n + 1
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Closure in decorator compiled")
            
        except Exception as e:
            self.fail(f"Closure in decorator failed: {e}")
    
    def test_returning_closure(self):
        """Test function that returns a closure"""
        source = """
def make_counter(start: int):
    count: int = start
    
    def increment() -> int:
        # Note: Modifying captured variables requires special handling
        # For now, just return the captured value
        return count
    
    return increment
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Returning closure compiled")
            
        except Exception as e:
            self.fail(f"Returning closure failed: {e}")
    
    def test_closure_with_function_call(self):
        """Test closure that calls another function"""
        source = """
def add(a: int, b: int) -> int:
    return a + b

def make_adder(n: int):
    def add_n(x: int) -> int:
        return add(x, n)  # Calls outer function with captured variable
    return add_n
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Closure with function call compiled")
            
        except Exception as e:
            self.fail(f"Closure with function call failed: {e}")


if __name__ == '__main__':
    unittest.main()
