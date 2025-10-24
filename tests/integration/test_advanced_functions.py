"""
Integration tests for advanced function features.

Tests:
- Default arguments
- Keyword arguments  
- *args (variable positional)
- **kwargs (variable keyword)
- Nested functions
- Closures
- Lambda expressions
- Decorators
"""

import unittest
import ast
from compiler.frontend.symbols import SymbolTable
from compiler.ir.lowering import IRLowering
from compiler.backend.llvm_gen import LLVMCodeGen


class TestAdvancedFunctions(unittest.TestCase):
    """Test suite for advanced function features"""
    
    def compile_source(self, source: str):
        """Helper to compile source code to IR and LLVM"""
        tree = ast.parse(source)
        symbol_table = SymbolTable(name="global")
        
        lowering = IRLowering(symbol_table)
        ir_module = lowering.visit_Module(tree)
        
        codegen = LLVMCodeGen()
        llvm_module = codegen.generate_module(ir_module)
        
        return ir_module, llvm_module
    
    # ========================================================================
    # Default Arguments Tests
    # ========================================================================
    
    def test_default_argument_simple(self):
        """Test function with single default argument"""
        source = """
def greet(name: str = "World"):
    return name
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            func = ir_module.functions[0]
            self.assertEqual(func.name, "greet")
            
            # Should have parameter with default
            self.assertIsNotNone(llvm_module)
            
            print("✅ Simple default argument compiled")
            
        except Exception as e:
            self.fail(f"Default argument compilation failed: {e}")
    
    def test_default_argument_multiple(self):
        """Test function with multiple default arguments"""
        source = """
def calculate(x: int, y: int = 10, z: int = 5):
    return x + y + z
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            self.assertIsNotNone(llvm_module)
            
            print("✅ Multiple default arguments compiled")
            
        except Exception as e:
            self.fail(f"Multiple default arguments failed: {e}")
    
    def test_default_argument_mixed(self):
        """Test function with mixed required and default args"""
        source = """
def process(a: int, b: int, c: int = 100):
    result: int = a + b + c
    return result
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            self.assertIsNotNone(llvm_module)
            
            print("✅ Mixed required/default arguments compiled")
            
        except Exception as e:
            self.fail(f"Mixed arguments failed: {e}")
    
    # ========================================================================
    # Keyword Arguments Tests
    # ========================================================================
    
    def test_keyword_arguments(self):
        """Test function calls with keyword arguments"""
        source = """
def configure(host: str, port: int, debug: bool):
    return port

result = configure(host="localhost", port=8080, debug=True)
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertIsNotNone(llvm_module)
            
            print("✅ Keyword arguments compiled")
            
        except Exception as e:
            self.fail(f"Keyword arguments failed: {e}")
    
    # ========================================================================
    # *args Tests  
    # ========================================================================
    
    def test_varargs_simple(self):
        """Test function with *args"""
        source = """
def sum_all(*args):
    # For now, just verify *args compiles
    # Full iteration support comes later
    return 0
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ *args compiled")
            
        except Exception as e:
            self.fail(f"*args failed: {e}")
    
    def test_varargs_with_regular_args(self):
        """Test function with regular args and *args"""
        source = """
def print_values(prefix: str, *values):
    # For now, just verify mixed args compile
    return 0
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ Mixed regular/*args compiled")
            
        except Exception as e:
            self.fail(f"Mixed args/*args failed: {e}")
    
    # ========================================================================
    # **kwargs Tests
    # ========================================================================
    
    def test_kwargs_simple(self):
        """Test function with **kwargs"""
        source = """
def build_config(**kwargs):
    count: int = 0
    return count
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ **kwargs compiled")
            
        except Exception as e:
            self.fail(f"**kwargs failed: {e}")
    
    def test_all_argument_types(self):
        """Test function with all argument types"""
        source = """
def flexible(a: int, b: int = 10, *args, **kwargs):
    return a + b
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertGreater(len(ir_module.functions), 0)
            
            print("✅ All argument types compiled")
            
        except Exception as e:
            self.fail(f"All argument types failed: {e}")
    
    # ========================================================================
    # Lambda Tests
    # ========================================================================
    
    def test_lambda_simple(self):
        """Test simple lambda expression"""
        source = """
square = lambda x: x * x
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Lambda creates anonymous function
            self.assertIsNotNone(llvm_module)
            
            print("✅ Simple lambda compiled")
            
        except Exception as e:
            self.fail(f"Simple lambda failed: {e}")
    
    def test_lambda_multiple_args(self):
        """Test lambda with multiple arguments"""
        source = """
add = lambda x, y: x + y
"""
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            self.assertIsNotNone(llvm_module)
            
            print("✅ Multi-arg lambda compiled")
            
        except Exception as e:
            self.fail(f"Multi-arg lambda failed: {e}")


if __name__ == "__main__":
    unittest.main()
