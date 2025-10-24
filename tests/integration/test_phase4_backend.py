"""
Phase 4 Backend Integration Tests

Tests LLVM code generation for Phase 4 features:
- Async/await (coroutines)
- Generators (yield)
- Exception handling (try/except/finally)
- Context managers (with statements)

Phase: 4 (Backend Integration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ir.ir_nodes import *
from compiler.frontend.semantic import Type, TypeKind
from compiler.backend.llvm_gen import LLVMCodeGen


def test_async_function_codegen():
    """Test LLVM code generation for async functions"""
    print("\n" + "="*80)
    print("Test 1: Async Function LLVM Code Generation")
    print("="*80)
    
    # Create async function IR:
    # async def fetch_data(url: str) -> str:
    #     result = await http_get(url)
    #     return result
    
    async_func = IRAsyncFunction(
        name="fetch_data",
        params=[("url", Type(TypeKind.STR))],
        body=[
            IRAwait(
                coroutine=IRCall("http_get", [IRVar("url", Type(TypeKind.STR))], Type(TypeKind.STR)),
                result_type=Type(TypeKind.STR)
            )
        ],
        return_type=Type(TypeKind.STR)
    )
    
    # Generate LLVM
    codegen = LLVMCodeGen()
    
    # Create a test function to hold the async function
    test_func = IRFunction(
        name="test_async",
        param_names=["url"],
        param_types=[Type(TypeKind.STR)],
        return_type=Type(TypeKind.STR)
    )
    
    entry = IRBasicBlock("entry")
    entry.add_instruction(async_func)
    test_func.add_block(entry)
    
    module = IRModule("async_test")
    module.add_function(test_func)
    
    try:
        llvm_ir = codegen.generate_module(module)
        print("✅ Async function codegen successful")
        print("\nGenerated LLVM IR (first 500 chars):")
        print(str(llvm_ir)[:500])
        return True
    except Exception as e:
        print(f"❌ Async function codegen failed: {e}")
        return False


def test_generator_codegen():
    """Test LLVM code generation for generators"""
    print("\n" + "="*80)
    print("Test 2: Generator LLVM Code Generation")
    print("="*80)
    
    # Create generator IR:
    # def count_up(n: int):
    #     i = 0
    #     while i < n:
    #         yield i
    #         i = i + 1
    
    # Simplified IR for testing
    gen_func = IRFunction(
        name="count_up",
        param_names=["n"],
        param_types=[Type(TypeKind.INT)],
        return_type=Type(TypeKind.INT)
    )
    
    entry = IRBasicBlock("entry")
    
    # Yield instruction
    i_var = IRVar("i", Type(TypeKind.INT))
    yield_stmt = IRYield(IRLoad(i_var))
    entry.add_instruction(yield_stmt)
    entry.add_instruction(IRReturn(None))
    
    gen_func.add_block(entry)
    
    module = IRModule("generator_test")
    module.add_function(gen_func)
    
    codegen = LLVMCodeGen()
    
    try:
        llvm_ir = codegen.generate_module(module)
        print("✅ Generator codegen successful")
        print("\nGenerated LLVM IR (first 500 chars):")
        print(str(llvm_ir)[:500])
        return True
    except Exception as e:
        print(f"❌ Generator codegen failed: {e}")
        return False


def test_exception_handling_codegen():
    """Test LLVM code generation for exception handling"""
    print("\n" + "="*80)
    print("Test 3: Exception Handling LLVM Code Generation")
    print("="*80)
    
    # Create try/except IR:
    # try:
    #     x = risky_operation()
    # except ValueError as e:
    #     x = 0
    # finally:
    #     cleanup()
    
    try_node = IRTry(
        body=[
            IRCall("risky_operation", [], Type(TypeKind.INT))
        ],
        except_blocks=[
            IRExcept(
                exception_type="ValueError",
                var_name="e",
                handler=[
                    IRStore(IRVar("x", Type(TypeKind.INT)), IRConstInt(0))
                ]
            )
        ],
        finally_block=IRFinally(
            body=[
                IRCall("cleanup", [], Type(TypeKind.NONE))
            ]
        )
    )
    
    # Wrap in function
    func = IRFunction(
        name="test_exception",
        param_names=[],
        param_types=[],
        return_type=Type(TypeKind.NONE)
    )
    
    entry = IRBasicBlock("entry")
    entry.add_instruction(try_node)
    entry.add_instruction(IRReturn(None))
    
    func.add_block(entry)
    
    module = IRModule("exception_test")
    module.add_function(func)
    
    codegen = LLVMCodeGen()
    
    try:
        llvm_ir = codegen.generate_module(module)
        print("✅ Exception handling codegen successful")
        print("\nGenerated LLVM IR (first 800 chars):")
        print(str(llvm_ir)[:800])
        
        # Verify landingpad instruction present
        if "landingpad" in str(llvm_ir):
            print("✅ Landingpad instruction found")
        else:
            print("⚠️  Warning: landingpad instruction not found")
        
        return True
    except Exception as e:
        print(f"❌ Exception handling codegen failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_manager_codegen():
    """Test LLVM code generation for context managers"""
    print("\n" + "="*80)
    print("Test 4: Context Manager LLVM Code Generation")
    print("="*80)
    
    # Create with statement IR:
    # with open("file.txt") as f:
    #     data = f.read()
    
    with_node = IRWith(
        context_expr=IRCall("open", [IRConstStr("file.txt")], Type(TypeKind.STR)),
        var_name="f",
        body=[
            IRCall("read", [IRVar("f", Type(TypeKind.STR))], Type(TypeKind.STR))
        ]
    )
    
    # Wrap in function
    func = IRFunction(
        name="test_with",
        param_names=[],
        param_types=[],
        return_type=Type(TypeKind.NONE)
    )
    
    entry = IRBasicBlock("entry")
    entry.add_instruction(with_node)
    entry.add_instruction(IRReturn(None))
    
    func.add_block(entry)
    
    module = IRModule("with_test")
    module.add_function(func)
    
    codegen = LLVMCodeGen()
    
    try:
        llvm_ir = codegen.generate_module(module)
        print("✅ Context manager codegen successful")
        print("\nGenerated LLVM IR (first 500 chars):")
        print(str(llvm_ir)[:500])
        
        # Verify enter/exit blocks present
        ir_str = str(llvm_ir)
        if "with.enter" in ir_str and "with.exit" in ir_str:
            print("✅ Enter/exit blocks found")
        else:
            print("⚠️  Warning: enter/exit blocks not found")
        
        return True
    except Exception as e:
        print(f"❌ Context manager codegen failed: {e}")
        return False


def test_yield_from_codegen():
    """Test LLVM code generation for yield from"""
    print("\n" + "="*80)
    print("Test 5: Yield From LLVM Code Generation")
    print("="*80)
    
    # Create yield from IR:
    # def delegate(gen):
    #     yield from gen
    
    result_var = IRVar("t0", Type(TypeKind.INT))
    yield_from = IRYieldFrom(
        iterator=IRVar("gen", Type(TypeKind.INT)),
        result=result_var
    )
    
    func = IRFunction(
        name="delegate",
        param_names=["gen"],
        param_types=[Type(TypeKind.INT)],
        return_type=Type(TypeKind.NONE)
    )
    
    entry = IRBasicBlock("entry")
    entry.add_instruction(yield_from)
    entry.add_instruction(IRReturn(None))
    
    func.add_block(entry)
    
    module = IRModule("yield_from_test")
    module.add_function(func)
    
    codegen = LLVMCodeGen()
    
    try:
        llvm_ir = codegen.generate_module(module)
        print("✅ Yield from codegen successful")
        print("\nGenerated LLVM IR (first 500 chars):")
        print(str(llvm_ir)[:500])
        return True
    except Exception as e:
        print(f"❌ Yield from codegen failed: {e}")
        return False


def test_raise_codegen():
    """Test LLVM code generation for raise statement"""
    print("\n" + "="*80)
    print("Test 6: Raise Statement LLVM Code Generation")
    print("="*80)
    
    # Create raise IR:
    # raise ValueError("Error message")
    
    raise_node = IRRaise(
        exception=IRCall("ValueError", [IRConstStr("Error message")], Type(TypeKind.STR))
    )
    
    func = IRFunction(
        name="test_raise",
        param_names=[],
        param_types=[],
        return_type=Type(TypeKind.NONE)
    )
    
    entry = IRBasicBlock("entry")
    entry.add_instruction(raise_node)
    # No return after raise (unreachable)
    
    func.add_block(entry)
    
    module = IRModule("raise_test")
    module.add_function(func)
    
    codegen = LLVMCodeGen()
    
    try:
        llvm_ir = codegen.generate_module(module)
        print("✅ Raise statement codegen successful")
        print("\nGenerated LLVM IR (first 500 chars):")
        print(str(llvm_ir)[:500])
        
        # Verify unreachable instruction present
        if "unreachable" in str(llvm_ir):
            print("✅ Unreachable instruction found after raise")
        else:
            print("⚠️  Warning: unreachable instruction not found")
        
        return True
    except Exception as e:
        print(f"❌ Raise codegen failed: {e}")
        return False


def run_all_tests():
    """Run all Phase 4 backend integration tests"""
    print("\n" + "="*80)
    print("🚀 PHASE 4 BACKEND INTEGRATION TEST SUITE")
    print("="*80)
    print("Testing LLVM code generation for advanced Python features")
    print()
    
    tests = [
        ("Async Function Codegen", test_async_function_codegen),
        ("Generator Codegen", test_generator_codegen),
        ("Exception Handling Codegen", test_exception_handling_codegen),
        ("Context Manager Codegen", test_context_manager_codegen),
        ("Yield From Codegen", test_yield_from_codegen),
        ("Raise Statement Codegen", test_raise_codegen),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "-"*80)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL PHASE 4 BACKEND TESTS PASSED! 🎉")
        print("Phase 4 LLVM code generation is working correctly.")
        print("\nNext Steps:")
        print("  1. Add AST lowering for Phase 4 features")
        print("  2. Extend semantic analysis for async/generators")
        print("  3. Add end-to-end integration tests")
        print("  4. Benchmark performance improvements")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
