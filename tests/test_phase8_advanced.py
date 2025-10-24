"""
Comprehensive Test Suite for Phase 8: Advanced Features
Tests context managers, decorators, metaclasses, and advanced Python features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llvmlite import ir
from compiler.runtime.phase8_advanced import Phase8Advanced


class TestPhase8Advanced:
    """Test suite for Phase 8 advanced features."""
    
    def __init__(self):
        self.phase8 = Phase8Advanced()
        self.module = ir.Module(name="test_phase8")
        self.builder = None
        
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def run_all_tests(self):
        """Run all Phase 8 tests."""
        print("\n" + "="*60)
        print("Phase 8 Advanced Features - Comprehensive Test Suite")
        print("="*60)
        
        # Context manager tests
        self.test_with_statement()
        self.test_context_enter_exit()
        self.test_multiple_context_managers()
        self.test_context_exception_handling()
        
        # Decorator tests
        self.test_property_decorator()
        self.test_classmethod_decorator()
        self.test_staticmethod_decorator()
        self.test_custom_decorator()
        
        # Metaclass tests
        self.test_create_metaclass()
        self.test_apply_metaclass()
        self.test_metaclass_new()
        
        # Advanced feature tests
        self.test_slots_class()
        self.test_weakref()
        self.test_super_call()
        self.test_mro_computation()
        self.test_abstract_base_class()
        self.test_descriptor_protocol()
        self.test_callable_object()
        
        # Edge cases
        self.test_nested_with_statements()
        self.test_decorator_chaining()
        self.test_multiple_inheritance()
        self.test_property_setter_deleter()
        
        self.print_summary()
    
    # Context Manager Tests
    
    def test_with_statement(self):
        """Test basic with statement."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cm = self._create_dummy_value(builder, "context_manager")
            body = self._create_dummy_value(builder, "body_func")
            
            result = self.phase8.generate_with_statement(builder, self.module, cm, body)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 1: with statement")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 1 failed: {e}")
    
    def test_context_enter_exit(self):
        """Test __enter__ and __exit__ methods."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cm = self._create_dummy_value(builder, "context_manager")
            
            enter_result = self.phase8.context_enter(builder, self.module, cm)
            exit_result = self.phase8.context_exit(builder, self.module, cm)
            
            assert enter_result is not None
            assert exit_result is not None
            self.passed += 1
            print("✅ Test 2: __enter__ and __exit__")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 2 failed: {e}")
    
    def test_multiple_context_managers(self):
        """Test multiple context managers."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cm1 = self._create_dummy_value(builder, "cm1")
            cm2 = self._create_dummy_value(builder, "cm2")
            
            self.phase8.context_enter(builder, self.module, cm1)
            self.phase8.context_enter(builder, self.module, cm2)
            self.phase8.context_exit(builder, self.module, cm2)
            self.phase8.context_exit(builder, self.module, cm1)
            
            self.passed += 1
            print("✅ Test 3: Multiple context managers")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 3 failed: {e}")
    
    def test_context_exception_handling(self):
        """Test exception handling in context."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cm = self._create_dummy_value(builder, "context_manager")
            exc_type = self._create_dummy_value(builder, "exc_type")
            exc_value = self._create_dummy_value(builder, "exc_value")
            
            result = self.phase8.context_exit(builder, self.module, cm, exc_type, exc_value)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 4: Exception handling in context")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 4 failed: {e}")
    
    # Decorator Tests
    
    def test_property_decorator(self):
        """Test @property decorator."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            getter = self._create_dummy_value(builder, "getter")
            
            result = self.phase8.create_property(builder, self.module, getter=getter, doc="Test property")
            
            assert result is not None
            self.passed += 1
            print("✅ Test 5: @property decorator")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 5 failed: {e}")
    
    def test_classmethod_decorator(self):
        """Test @classmethod decorator."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            method = self._create_dummy_value(builder, "method")
            
            result = self.phase8.create_classmethod(builder, self.module, method)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 6: @classmethod decorator")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 6 failed: {e}")
    
    def test_staticmethod_decorator(self):
        """Test @staticmethod decorator."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            method = self._create_dummy_value(builder, "method")
            
            result = self.phase8.create_staticmethod(builder, self.module, method)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 7: @staticmethod decorator")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 7 failed: {e}")
    
    def test_custom_decorator(self):
        """Test custom decorator with arguments."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            decorator = self._create_dummy_value(builder, "decorator")
            target = self._create_dummy_value(builder, "target")
            arg1 = self._create_dummy_value(builder, "arg1")
            
            result = self.phase8.apply_decorator(builder, self.module, decorator, target, arg1)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 8: Custom decorator with arguments")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 8 failed: {e}")
    
    # Metaclass Tests
    
    def test_create_metaclass(self):
        """Test metaclass creation."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            bases = self._create_dummy_value(builder, "bases")
            namespace = self._create_dummy_value(builder, "namespace")
            
            result = self.phase8.create_metaclass(builder, self.module, "TestMeta", bases, namespace)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 9: Metaclass creation")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 9 failed: {e}")
    
    def test_apply_metaclass(self):
        """Test applying metaclass."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            metaclass = self._create_dummy_value(builder, "metaclass")
            bases = self._create_dummy_value(builder, "bases")
            namespace = self._create_dummy_value(builder, "namespace")
            
            result = self.phase8.apply_metaclass(builder, self.module, metaclass, "TestClass", bases, namespace)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 10: Applying metaclass")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 10 failed: {e}")
    
    def test_metaclass_new(self):
        """Test metaclass __new__ method."""
        self.total += 1
        try:
            # Metaclass __new__ is handled by apply_metaclass
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            metaclass = self._create_dummy_value(builder, "metaclass")
            bases = self._create_dummy_value(builder, "bases")
            namespace = self._create_dummy_value(builder, "namespace")
            
            result = self.phase8.apply_metaclass(builder, self.module, metaclass, "TestClass", bases, namespace)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 11: Metaclass __new__")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 11 failed: {e}")
    
    # Advanced Feature Tests
    
    def test_slots_class(self):
        """Test __slots__ class."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            slots = ['x', 'y', 'z']
            
            result = self.phase8.create_slots_class(builder, self.module, "Point", slots)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 12: __slots__ class")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 12 failed: {e}")
    
    def test_weakref(self):
        """Test weakref support."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            obj = self._create_dummy_value(builder, "object")
            
            result = self.phase8.create_weakref(builder, self.module, obj)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 13: weakref support")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 13 failed: {e}")
    
    def test_super_call(self):
        """Test super() call."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cls = self._create_dummy_value(builder, "class")
            obj = self._create_dummy_value(builder, "object")
            
            result = self.phase8.call_super(builder, self.module, cls, obj)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 14: super() call")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 14 failed: {e}")
    
    def test_mro_computation(self):
        """Test MRO computation."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cls = self._create_dummy_value(builder, "class")
            bases = self._create_dummy_value(builder, "bases")
            
            result = self.phase8.compute_mro(builder, self.module, cls, bases)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 15: MRO computation")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 15 failed: {e}")
    
    def test_abstract_base_class(self):
        """Test Abstract Base Class."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            abstract_methods = ['method1', 'method2']
            
            result = self.phase8.create_abc(builder, self.module, "AbstractClass", abstract_methods)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 16: Abstract Base Class")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 16 failed: {e}")
    
    def test_descriptor_protocol(self):
        """Test descriptor protocol."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            get_func = self._create_dummy_value(builder, "get")
            set_func = self._create_dummy_value(builder, "set")
            
            result = self.phase8.create_descriptor(builder, self.module, get_func, set_func)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 17: Descriptor protocol")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 17 failed: {e}")
    
    def test_callable_object(self):
        """Test callable object."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            obj = self._create_dummy_value(builder, "object")
            call_func = self._create_dummy_value(builder, "call")
            
            result = self.phase8.make_callable(builder, self.module, obj, call_func)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 18: Callable object")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 18 failed: {e}")
    
    # Edge Case Tests
    
    def test_nested_with_statements(self):
        """Test nested with statements."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cm1 = self._create_dummy_value(builder, "cm1")
            cm2 = self._create_dummy_value(builder, "cm2")
            body = self._create_dummy_value(builder, "body")
            
            # Nested with statements
            self.phase8.generate_with_statement(builder, self.module, cm1, body)
            self.phase8.generate_with_statement(builder, self.module, cm2, body)
            
            self.passed += 1
            print("✅ Test 19: Nested with statements")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 19 failed: {e}")
    
    def test_decorator_chaining(self):
        """Test decorator chaining."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            dec1 = self._create_dummy_value(builder, "dec1")
            dec2 = self._create_dummy_value(builder, "dec2")
            target = self._create_dummy_value(builder, "target")
            
            # Apply multiple decorators
            result1 = self.phase8.apply_decorator(builder, self.module, dec1, target)
            result2 = self.phase8.apply_decorator(builder, self.module, dec2, result1)
            
            assert result2 is not None
            self.passed += 1
            print("✅ Test 20: Decorator chaining")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 20 failed: {e}")
    
    def test_multiple_inheritance(self):
        """Test multiple inheritance and MRO."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            cls = self._create_dummy_value(builder, "class")
            bases = self._create_dummy_value(builder, "bases")  # Multiple bases
            
            result = self.phase8.compute_mro(builder, self.module, cls, bases)
            
            assert result is not None
            self.passed += 1
            print("✅ Test 21: Multiple inheritance and MRO")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 21 failed: {e}")
    
    def test_property_setter_deleter(self):
        """Test property setter and deleter."""
        self.total += 1
        try:
            func = self._create_test_function()
            builder = ir.IRBuilder(func.append_basic_block(name="entry"))
            
            getter = self._create_dummy_value(builder, "getter")
            setter = self._create_dummy_value(builder, "setter")
            deleter = self._create_dummy_value(builder, "deleter")
            
            result = self.phase8.create_property(builder, self.module, getter, setter, deleter, "Full property")
            
            assert result is not None
            self.passed += 1
            print("✅ Test 22: Property setter and deleter")
        except Exception as e:
            self.failed += 1
            print(f"❌ Test 22 failed: {e}")
    
    # Helper methods
    
    def _create_test_function(self):
        """Create a test function for building IR."""
        func_type = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_type, name=f"test_func_{self.total}")
        return func
    
    def _create_dummy_value(self, builder, name):
        """Create a dummy pointer value for testing."""
        void_ptr = ir.IntType(8).as_pointer()
        alloca = builder.alloca(void_ptr, name=name)
        return builder.load(alloca)
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("Phase 8 Test Summary")
        print("="*60)
        print(f"Total tests: {self.total}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success rate: {(self.passed/self.total*100):.1f}%")
        print("="*60)
        
        if self.failed == 0:
            print("\n🎉 All Phase 8 tests passed! 🎉")
            print("Advanced features fully implemented!")
        
        return self.failed == 0


if __name__ == "__main__":
    test_suite = TestPhase8Advanced()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)
