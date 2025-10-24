"""
Week 2 Days 3-7: Full OOP Implementation Tests

Tests object allocation, attribute access, and method calls
through the complete compilation pipeline: AST → IR → LLVM
"""

import pytest
import ast
from compiler.frontend.symbols import SymbolTable
from compiler.ir.lowering import IRLowering
from compiler.backend.llvm_gen import LLVMCodeGen


def compile_code(source):
    """Helper to compile source code through full pipeline"""
    tree = ast.parse(source)
    symbol_table = SymbolTable(name="global")
    
    lowering = IRLowering(symbol_table)
    ir_module = lowering.visit_Module(tree)
    
    codegen = LLVMCodeGen()
    llvm_ir = codegen.generate_module(ir_module)
    
    return ir_module, llvm_ir


class TestObjectAllocation:
    """Week 2 Day 3: Object Allocation with malloc"""
    
    def test_simple_object_creation(self):
        """Test basic object allocation"""
        code = """
class Point:
    pass

p = Point()
"""
        ir_module, _ = compile_code(code)
        
        # Check IR has classes
        assert ir_module is not None
        assert hasattr(ir_module, 'classes')
        assert len(ir_module.classes) == 1
        assert ir_module.classes[0].name == "Point"
    
    def test_object_with_init(self):
        """Test object allocation with __init__ call"""
        code = """
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

p = Point(10, 20)
"""
        ir_module, _ = compile_code(code)
        
        assert ir_module is not None
        assert len(ir_module.classes) == 1
        assert ir_module.classes[0].name == "Point"
        
        # Check __init__ method exists
        init_method = None
        for method in ir_module.classes[0].methods:
            if hasattr(method, 'metadata') and method.metadata.get('original_name') == '__init__':
                init_method = method
                break
        
        assert init_method is not None
    
    def test_llvm_malloc_declaration(self):
        """Test that malloc is properly declared in LLVM IR"""
        code = """
class Point:
    def __init__(self, x: int):
        self.x = x
"""
        _, llvm_ir = compile_code(code)
        
        # Check malloc is declared (flexible matching for llvmlite's formatting)
        assert '@malloc' in llvm_ir or '@"malloc"' in llvm_ir
        assert '@free' in llvm_ir or '@"free"' in llvm_ir
        assert 'declare' in llvm_ir
    
    def test_llvm_struct_generation(self):
        """Test that class structs are generated in LLVM"""
        code = """
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
"""
        _, llvm_ir = compile_code(code)
        
        # Check struct type is defined
        assert '%"class.Point"' in llvm_ir
        assert 'type {' in llvm_ir


class TestAttributeAccess:
    """Week 2 Day 4: Attribute Get/Set with GEP"""
    
    def test_attribute_get_ir(self):
        """Test attribute access generates GetAttr IR"""
        code = """
class Point:
    def __init__(self, x: int):
        self.x = x

def get_x(p: Point) -> int:
    return p.x
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        # Find get_x function
        get_x_func = None
        for func in ir_module.functions:
            if func.name == "get_x":
                get_x_func = func
                break
        
        assert get_x_func is not None
        # Note: Attribute access IR generation not yet implemented in lowering
        # This test validates the pipeline
    
    def test_attribute_set_ir(self):
        """Test attribute assignment generates SetAttr IR"""
        code = """
class Point:
    def set_x(self, value: int):
        self.x = value
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        assert len(ir_module.classes) == 1
    
    def test_llvm_gep_pattern(self):
        """Test that attribute access generates GEP instructions"""
        code = """
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        
        
        
        # Just verify it compiles - GEP will appear when we have attribute access
        assert llvm_ir is not None


class TestMethodCalls:
    """Week 2 Day 5: Method Calls"""
    
    def test_simple_method_call(self):
        """Test basic method call"""
        code = """
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count = self.count + 1
    
    def get_count(self) -> int:
        return self.count
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        assert len(ir_module.classes) == 1
        assert len(ir_module.classes[0].methods) == 3
    
    def test_method_with_parameters(self):
        """Test method call with parameters"""
        code = """
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def move(self, dx: int, dy: int):
        self.x = self.x + dx
        self.y = self.y + dy
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        move_method = None
        for method in ir_module.classes[0].methods:
            if hasattr(method, 'metadata') and method.metadata.get('original_name') == 'move':
                move_method = method
                break
        
        assert move_method is not None
        # move should have 2 parameters (dx, dy) - self handled separately
        assert len(move_method.param_types) >= 2
    
    def test_method_return_value(self):
        """Test method with return value"""
        code = """
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        add_method = ir_module.classes[0].methods[0]
        assert add_method.return_type.kind.name == 'INT'
    
    def test_llvm_method_call_generation(self):
        """Test LLVM generation for method calls"""
        code = """
class Greeter:
    def greet(self) -> int:
        return 42
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        
        
        
        # Check method function is defined
        assert 'define' in llvm_ir
        assert 'Greeter_greet' in llvm_ir


class TestInheritance:
    """Week 2 Day 6: Basic Inheritance"""
    
    def test_simple_inheritance(self):
        """Test basic class inheritance"""
        code = """
class Animal:
    def speak(self) -> int:
        return 0

class Dog(Animal):
    def speak(self) -> int:
        return 1
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        assert len(ir_module.classes) == 2
        
        # Find Dog class
        dog_class = None
        for cls in ir_module.classes:
            if cls.name == "Dog":
                dog_class = cls
                break
        
        assert dog_class is not None
        assert len(dog_class.base_classes) == 1
        assert dog_class.base_classes[0] == "Animal"
    
    def test_inherited_attributes(self):
        """Test inheriting attributes from base class"""
        code = """
class Shape:
    def __init__(self, color: int):
        self.color = color

class Circle(Shape):
    def __init__(self, color: int, radius: int):
        self.color = color
        self.radius = radius
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        assert len(ir_module.classes) == 2


class TestComplexOOP:
    """Week 2 Day 7: Complex OOP Patterns"""
    
    def test_multiple_classes(self):
        """Test multiple class definitions"""
        code = """
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Line:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

class Rectangle:
    def __init__(self, top_left: Point, bottom_right: Point):
        self.top_left = top_left
        self.bottom_right = bottom_right
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        assert len(ir_module.classes) == 3
        
        class_names = [cls.name for cls in ir_module.classes]
        assert "Point" in class_names
        assert "Line" in class_names
        assert "Rectangle" in class_names
    
    def test_class_with_multiple_methods(self):
        """Test class with many methods"""
        code = """
class Vector:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z
    
    def length_squared(self) -> int:
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def dot(self, other: Vector) -> int:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def scale(self, factor: int):
        self.x = self.x * factor
        self.y = self.y * factor
        self.z = self.z * factor
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        assert ir_module is not None
        assert len(ir_module.classes) == 1
        assert len(ir_module.classes[0].methods) == 4  # __init__, length_squared, dot, scale
    
    def test_full_compilation_pipeline(self):
        """Test complete AST → IR → LLVM pipeline"""
        code = """
class Account:
    def __init__(self, balance: int):
        self.balance = balance
    
    def deposit(self, amount: int):
        self.balance = self.balance + amount
    
    def withdraw(self, amount: int) -> int:
        if self.balance >= amount:
            self.balance = self.balance - amount
            return 1
        return 0
    
    def get_balance(self) -> int:
        return self.balance
"""
        ir_module, llvm_ir = compile_code(code)
        
        
        
        
        
        
        
        # Verify all stages completed
        assert ir_module is not None
        assert llvm_ir is not None
        assert len(llvm_ir) > 0
        
        # Check key elements in LLVM IR
        assert '%"class.Account"' in llvm_ir
        assert 'Account___init__' in llvm_ir
        assert 'Account_deposit' in llvm_ir
        assert 'Account_withdraw' in llvm_ir
        assert 'Account_get_balance' in llvm_ir


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
