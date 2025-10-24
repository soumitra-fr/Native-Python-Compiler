"""
Test LLVM struct generation for classes
"""

import ast
from compiler.frontend.symbols import SymbolTable
from compiler.ir.lowering import IRLowering
from compiler.backend.llvm_gen import LLVMCodeGen


def test_class_struct_generation():
    """Test that classes generate LLVM struct types"""
    source = """
class Point:
    x: int
    y: int
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

def test() -> int:
    return 0
"""
    
    # Compile to IR
    tree = ast.parse(source)
    symbol_table = SymbolTable(name="global")
    
    lowering = IRLowering(symbol_table)
    ir_module = lowering.visit_Module(tree)
    
    # Generate LLVM
    codegen = LLVMCodeGen()
    llvm_ir = codegen.generate_module(ir_module)
    
    # Check that struct type was created
    assert "class.Point" in llvm_ir
    print("✅ Class struct type generated!")
    print("\nLLVM IR excerpt:")
    for line in llvm_ir.split('\n')[:20]:
        if 'class.Point' in line or '%' in line:
            print(line)


def test_empty_class_struct():
    """Test empty class generates valid struct"""
    source = """
class Empty:
    pass

def test() -> int:
    return 0
"""
    
    tree = ast.parse(source)
    symbol_table = SymbolTable(name="global")
    
    lowering = IRLowering(symbol_table)
    ir_module = lowering.visit_Module(tree)
    
    codegen = LLVMCodeGen()
    llvm_ir = codegen.generate_module(ir_module)
    
    # Empty class should have dummy field
    assert "class.Empty" in llvm_ir
    print("\n✅ Empty class struct generated!")


def test_inheritance_struct():
    """Test class with inheritance"""
    source = """
class Base:
    value: int

class Derived(Base):
    extra: int

def test() -> int:
    return 0
"""
    
    tree = ast.parse(source)
    symbol_table = SymbolTable(name="global")
    
    lowering = IRLowering(symbol_table)
    ir_module = lowering.visit_Module(tree)
    
    codegen = LLVMCodeGen()
    llvm_ir = codegen.generate_module(ir_module)
    
    # Both classes should generate structs
    assert "class.Base" in llvm_ir
    assert "class.Derived" in llvm_ir
    print("\n✅ Inheritance structs generated!")


if __name__ == '__main__':
    test_class_struct_generation()
    test_empty_class_struct()
    test_inheritance_struct()
    print("\n🎉 All LLVM struct generation tests passed!")
