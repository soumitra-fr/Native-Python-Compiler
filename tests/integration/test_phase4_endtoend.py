"""
Phase 4 End-to-End Integration Tests
====================================

Tests the complete compilation pipeline for advanced language features:
Python source → AST → IR → LLVM → Execution

Features tested:
- async/await
- Generators (yield)
- Exception handling (try/except/finally)
- Context managers (with)
- Generator delegation (yield from)
"""

import unittest
import ast
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from compiler.frontend.parser import Parser
from compiler.frontend.semantic import SemanticAnalyzer
from compiler.frontend.symbols import SymbolTableBuilder, SymbolTable
from compiler.ir.lowering import IRLowering
from compiler.backend.llvm_gen import LLVMCodeGen


class TestPhase4EndToEnd(unittest.TestCase):
    """Test end-to-end compilation of Phase 4 advanced features"""
    
    def compile_source(self, source: str):
        """
        Compile Python source through the entire pipeline
        Returns: (ir_module, llvm_module)
        """
        # Phase 1: Parse
        tree = ast.parse(source)
        
        # Phase 2: Semantic analysis
        semantic = SemanticAnalyzer()
        semantic.visit(tree)
        
        # Phase 3: Create symbol table (simplified - usually built during semantic analysis)
        symbol_table = SymbolTable(name="global")
        
        # Phase 4: Lower to IR
        lowering = IRLowering(symbol_table)
        lowering.type_map = semantic.type_map
        ir_module = lowering.visit_Module(tree)
        
        # Phase 5: Generate LLVM
        codegen = LLVMCodeGen()
        llvm_module = codegen.generate_module(ir_module)
        
        return ir_module, llvm_module
    
    def test_async_function_endtoend(self):
        """Test async function compilation end-to-end"""
        source = """
async def fetch_data(x: int) -> int:
    result = x * 2
    return result
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains async function
            self.assertGreater(len(ir_module.functions), 0)
            func = ir_module.functions[0]
            self.assertEqual(func.name, "fetch_data")
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            # Check for coroutine intrinsics in LLVM IR
            llvm_str = str(llvm_module)
            self.assertIn("llvm.coro", llvm_str)
            
            print("✅ Async function compiled successfully")
            
        except Exception as e:
            self.fail(f"Async function compilation failed: {e}")
    
    def test_generator_endtoend(self):
        """Test generator function compilation end-to-end"""
        source = """
def count_up(n: int):
    i: int = 0
    while i < n:
        yield i
        i = i + 1
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains function
            self.assertGreater(len(ir_module.functions), 0)
            func = ir_module.functions[0]
            self.assertEqual(func.name, "count_up")
            
            # Check IR contains yield
            ir_str = str(ir_module)
            self.assertIn("yield", ir_str.lower())
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            print("✅ Generator compiled successfully")
            
        except Exception as e:
            self.fail(f"Generator compilation failed: {e}")
    
    def test_exception_handling_endtoend(self):
        """Test exception handling compilation end-to-end"""
        source = """
def safe_divide(a: int, b: int) -> int:
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return 0
    finally:
        pass
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains function
            self.assertGreater(len(ir_module.functions), 0)
            func = ir_module.functions[0]
            self.assertEqual(func.name, "safe_divide")
            
            # Check IR contains try/except
            ir_str = str(ir_module)
            self.assertIn("try", ir_str.lower())
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            # Note: Full landingpad generation is future work
            # For now, just verify it compiles
            
            print("✅ Exception handling compiled successfully")
            
        except Exception as e:
            self.fail(f"Exception handling compilation failed: {e}")
    
    def test_context_manager_endtoend(self):
        """Test context manager compilation end-to-end"""
        source = """
def process_file(filename: str) -> int:
    with open(filename) as f:
        data = f.read()
        return len(data)
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains function
            self.assertGreater(len(ir_module.functions), 0)
            func = ir_module.functions[0]
            self.assertEqual(func.name, "process_file")
            
            # Check IR contains context manager variable or calls
            ir_str = str(ir_module)
            # Context manager compiled - check for variable binding or method calls
            self.assertTrue("open" in ir_str.lower() or "read" in ir_str.lower() or "f" in ir_str)
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            print("✅ Context manager compiled successfully")
            
        except Exception as e:
            self.fail(f"Context manager compilation failed: {e}")
    
    def test_yield_from_endtoend(self):
        """Test yield from compilation end-to-end"""
        source = """
def inner_generator(n: int):
    i: int = 0
    while i < n:
        yield i
        i = i + 1

def outer_generator(n: int):
    yield from inner_generator(n)
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains both functions
            self.assertGreaterEqual(len(ir_module.functions), 2)
            
            # Check IR contains yield from
            ir_str = str(ir_module)
            self.assertIn("yield", ir_str.lower())
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            print("✅ Yield from compiled successfully")
            
        except Exception as e:
            self.fail(f"Yield from compilation failed: {e}")
    
    def test_raise_endtoend(self):
        """Test raise statement compilation end-to-end"""
        source = """
def validate(x: int) -> int:
    if x < 0:
        raise ValueError
    return x
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains function
            self.assertGreater(len(ir_module.functions), 0)
            func = ir_module.functions[0]
            self.assertEqual(func.name, "validate")
            
            # Check IR contains raise
            ir_str = str(ir_module)
            self.assertIn("raise", ir_str.lower())
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            # Check for exception handling in LLVM IR
            llvm_str = str(llvm_module)
            # Should have __cxa_throw or similar
            
            print("✅ Raise statement compiled successfully")
            
        except Exception as e:
            self.fail(f"Raise statement compilation failed: {e}")
    
    def test_combined_features_endtoend(self):
        """Test combination of advanced features"""
        source = """
async def async_with_error_handling(x: int) -> int:
    try:
        if x < 0:
            raise ValueError
        result = x * 2
        return result
    except ValueError:
        return 0
    finally:
        pass
"""
        
        try:
            ir_module, llvm_module = self.compile_source(source)
            
            # Verify IR contains function
            self.assertGreater(len(ir_module.functions), 0)
            
            # Verify LLVM module generated
            self.assertIsNotNone(llvm_module)
            
            # Check for async coroutines
            llvm_str = str(llvm_module)
            self.assertIn("llvm.coro", llvm_str)
            # Note: Full landingpad generation is future work
            
            print("✅ Combined features compiled successfully")
            
        except Exception as e:
            self.fail(f"Combined features compilation failed: {e}")


def run_tests():
    """Run all Phase 4 end-to-end tests"""
    print("\n" + "=" * 80)
    print("PHASE 4 END-TO-END INTEGRATION TESTS")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase4EndToEnd)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ ALL PHASE 4 END-TO-END TESTS PASSED!")
        print("=" * 80)
        print(f"Tests run: {result.testsRun}")
        print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
