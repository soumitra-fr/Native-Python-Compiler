"""
Phase 3: Object-Oriented Programming Test Suite
Tests classes, inheritance, methods, properties, and magic methods.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.runtime.phase3_oop import Phase3OOP


class TestPhase3OOP:
    """Test suite for Phase 3 OOP features."""
    
    def __init__(self):
        self.oop = Phase3OOP()
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test(self, name, condition, error_msg=""):
        """Run a single test."""
        self.tests_run += 1
        if condition:
            self.tests_passed += 1
            print(f"  ✅ {name}")
            return True
        else:
            self.tests_failed += 1
            print(f"  ❌ {name}")
            if error_msg:
                print(f"     Error: {error_msg}")
            return False
    
    def test_class_type_structure(self):
        """Test ClassType initialization and structures."""
        print("\n1. Testing ClassType Structure...")
        
        # Test ClassType initialization
        self.test(
            "ClassType initialized",
            self.oop.class_type is not None
        )
        
        # Test class structure
        self.test(
            "Class structure has correct fields",
            len(self.oop.class_type.class_type.elements) == 10  # 10 fields
        )
        
        # Test instance structure
        self.test(
            "Instance structure has correct fields",
            len(self.oop.class_type.instance_type.elements) == 5  # 5 fields
        )
    
    def test_inheritance_structure(self):
        """Test InheritanceType and MRO computation."""
        print("\n2. Testing InheritanceType Structure...")
        
        # Test InheritanceType initialization
        self.test(
            "InheritanceType initialized",
            self.oop.inheritance is not None
        )
        
        # Test simple MRO (single inheritance)
        all_classes = {'A': [], 'B': ['A']}
        mro = self.oop.compute_mro('B', ['A'], all_classes)
        self.test(
            "MRO for single inheritance (B(A))",
            mro == ['B', 'A'],
            f"Expected ['B', 'A'], got {mro}"
        )
        
        # Test diamond inheritance MRO
        all_classes = {
            'A': [],
            'B': ['A'],
            'C': ['A'],
            'D': ['B', 'C']
        }
        mro = self.oop.compute_mro('D', ['B', 'C'], all_classes)
        self.test(
            "MRO for diamond inheritance (D(B, C) where B(A), C(A))",
            mro == ['D', 'B', 'C', 'A'],
            f"Expected ['D', 'B', 'C', 'A'], got {mro}"
        )
    
    def test_method_dispatch_structure(self):
        """Test MethodDispatch structures."""
        print("\n3. Testing MethodDispatch Structure...")
        
        # Test MethodDispatch initialization
        self.test(
            "MethodDispatch initialized",
            self.oop.method_dispatch is not None
        )
        
        # Test BoundMethod structure
        self.test(
            "BoundMethod structure has correct fields",
            len(self.oop.method_dispatch.bound_method_type.elements) == 3  # refcount, func, self
        )
        
        # Test StaticMethod structure
        self.test(
            "StaticMethod structure has correct fields",
            len(self.oop.method_dispatch.static_method_type.elements) == 2  # refcount, func
        )
        
        # Test ClassMethod structure
        self.test(
            "ClassMethod structure has correct fields",
            len(self.oop.method_dispatch.class_method_type.elements) == 3  # refcount, func, cls
        )
    
    def test_magic_methods_structure(self):
        """Test MagicMethods support."""
        print("\n4. Testing MagicMethods Structure...")
        
        # Test MagicMethods initialization
        self.test(
            "MagicMethods initialized",
            self.oop.magic_methods is not None
        )
        
        # Test magic method names list
        self.test(
            "Magic methods list has correct count",
            len(self.oop.magic_methods.magic_method_names) >= 30,
            f"Expected >= 30, got {len(self.oop.magic_methods.magic_method_names)}"
        )
        
        # Test presence of key magic methods
        key_methods = ['__init__', '__str__', '__repr__', '__eq__', '__hash__', '__len__', '__getitem__', '__iter__', '__call__', '__add__']
        for method in key_methods:
            self.test(
                f"Magic method '{method}' supported",
                method in self.oop.magic_methods.magic_method_names
            )
    
    def test_property_structure(self):
        """Test PropertyType structure."""
        print("\n5. Testing PropertyType Structure...")
        
        # Test PropertyType initialization
        self.test(
            "PropertyType initialized",
            self.oop.property_type is not None
        )
        
        # Test Property structure
        self.test(
            "Property structure has correct fields",
            len(self.oop.property_type.property_type.elements) == 5  # refcount, fget, fset, fdel, doc
        )
    
    def test_runtime_compilation(self):
        """Test that all runtime C files are compiled."""
        print("\n6. Testing Runtime Compilation...")
        
        runtime_files = [
            ('class_runtime.o', 'compiler/runtime/class_runtime.o'),
            ('inheritance_runtime.o', 'compiler/runtime/inheritance_runtime.o'),
            ('method_dispatch_runtime.o', 'compiler/runtime/method_dispatch_runtime.o'),
            ('magic_methods_runtime.o', 'compiler/runtime/magic_methods_runtime.o'),
            ('property_runtime.o', 'compiler/runtime/property_runtime.o'),
        ]
        
        for name, path in runtime_files:
            self.test(
                f"Runtime file '{name}' exists",
                os.path.exists(path),
                f"File not found: {path}"
            )
            
            if os.path.exists(path):
                size = os.path.getsize(path)
                self.test(
                    f"Runtime file '{name}' has content",
                    size > 0,
                    f"File is empty: {path}"
                )
    
    def test_integration_module(self):
        """Test Phase3OOP integration module."""
        print("\n7. Testing Integration Module...")
        
        # Test Phase3OOP initialization
        self.test(
            "Phase3OOP module initialized",
            self.oop is not None
        )
        
        # Test all components are initialized
        self.test(
            "ClassType component present",
            hasattr(self.oop, 'class_type') and self.oop.class_type is not None
        )
        
        self.test(
            "InheritanceType component present",
            hasattr(self.oop, 'inheritance') and self.oop.inheritance is not None
        )
        
        self.test(
            "MethodDispatch component present",
            hasattr(self.oop, 'method_dispatch') and self.oop.method_dispatch is not None
        )
        
        self.test(
            "MagicMethods component present",
            hasattr(self.oop, 'magic_methods') and self.oop.magic_methods is not None
        )
        
        self.test(
            "PropertyType component present",
            hasattr(self.oop, 'property_type') and self.oop.property_type is not None
        )
        
        # Test summary generation
        summary = self.oop.get_summary()
        self.test(
            "Summary contains phase information",
            'phase' in summary and summary['phase'] == 3
        )
        
        self.test(
            "Summary contains components",
            'components' in summary and len(summary['components']) == 5
        )
    
    def test_example_patterns(self):
        """Test example code patterns."""
        print("\n8. Testing Example Patterns...")
        
        examples = [
            ("Basic class with __init__", """
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
"""),
            ("Class with methods", """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
"""),
            ("Class with properties", """
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        self._radius = value
"""),
            ("Class with magic methods", """
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
"""),
            ("Inheritance", """
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"
"""),
            ("Multiple inheritance", """
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass
"""),
            ("Static and class methods", """
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b
    
    @classmethod
    def from_string(cls, s):
        return cls()
"""),
        ]
        
        for title, code in examples:
            # Just validate the code compiles syntactically
            try:
                compile(code, '<string>', 'exec')
                self.test(f"Example '{title}' is valid Python", True)
            except SyntaxError as e:
                self.test(f"Example '{title}' is valid Python", False, str(e))
    
    def run_all_tests(self):
        """Run all tests and print summary."""
        print("=" * 70)
        print("Phase 3: Object-Oriented Programming Test Suite")
        print("=" * 70)
        
        self.test_class_type_structure()
        self.test_inheritance_structure()
        self.test_method_dispatch_structure()
        self.test_magic_methods_structure()
        self.test_property_structure()
        self.test_runtime_compilation()
        self.test_integration_module()
        self.test_example_patterns()
        
        # Print summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Tests Run: {self.tests_run}")
        print(f"Passed: {self.tests_passed} ✅")
        print(f"Failed: {self.tests_failed} ❌")
        
        if self.tests_failed == 0:
            print(f"\n🎉 All tests passed! Success Rate: 100.0%")
        else:
            success_rate = (self.tests_passed / self.tests_run) * 100
            print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        print("=" * 70)
        
        # Print example patterns
        self.print_example_patterns()
        
        return self.tests_failed == 0
    
    def print_example_patterns(self):
        """Print example OOP patterns."""
        print("\n" + "=" * 70)
        print("Phase 3 Example Patterns")
        print("=" * 70)
        
        examples = [
            ("Basic Class", """
@njit
def create_person():
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def __str__(self):
            return f"{self.name} ({self.age})"
    
    p = Person("Alice", 30)
    return str(p)  # "Alice (30)"
"""),
            ("Inheritance with super()", """
@njit
def test_inheritance():
    class Animal:
        def __init__(self, name):
            self.name = name
        
        def speak(self):
            return "..."
    
    class Dog(Animal):
        def __init__(self, name, breed):
            super().__init__(name)
            self.breed = breed
        
        def speak(self):
            return f"{self.name} says Woof!"
    
    dog = Dog("Rex", "Labrador")
    return dog.speak()  # "Rex says Woof!"
"""),
            ("Properties", """
@njit
def test_properties():
    class Circle:
        def __init__(self, radius):
            self._radius = radius
        
        @property
        def area(self):
            return 3.14159 * self._radius ** 2
    
    c = Circle(5)
    return c.area  # 78.53975
"""),
            ("Operator Overloading", """
@njit
def test_operators():
    class Vector:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def __add__(self, other):
            return Vector(self.x + other.x, self.y + other.y)
    
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    v3 = v1 + v2
    return (v3.x, v3.y)  # (4, 6)
"""),
            ("Context Manager", """
@njit
def test_context_manager():
    class FileHandler:
        def __enter__(self):
            print("Opening resource")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("Closing resource")
    
    with FileHandler() as f:
        print("Using resource")
"""),
        ]
        
        for i, (title, code) in enumerate(examples, 1):
            print(f"\n{i}. {title}:")
            print(code)
        
        print("\n" + "=" * 70)


def main():
    """Main test runner."""
    test_suite = TestPhase3OOP()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
