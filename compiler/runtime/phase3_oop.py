"""
Phase 3: Object-Oriented Programming Integration Module
Provides unified API for all OOP features.
"""

import os
from .class_type import ClassType
from .inheritance import InheritanceType
from .method_dispatch import MethodDispatch
from .magic_methods import MagicMethods
from .property_type import PropertyType

class Phase3OOP:
    """
    Integration module for Phase 3: Object-Oriented Programming.
    
    Combines:
    - Classes and instances
    - Inheritance and MRO
    - Method dispatch (bound/static/class methods)
    - Magic methods (dunder methods)
    - Properties and descriptors
    """
    
    def __init__(self):
        # Initialize all OOP components
        self.class_type = ClassType()
        self.inheritance = InheritanceType()
        self.method_dispatch = MethodDispatch()
        self.magic_methods = MagicMethods()
        self.property_type = PropertyType()
        
        # Runtime object files
        self.runtime_objects = [
            'compiler/runtime/class_runtime.o',
            'compiler/runtime/inheritance_runtime.o',
            'compiler/runtime/method_dispatch_runtime.o',
            'compiler/runtime/magic_methods_runtime.o',
            'compiler/runtime/property_runtime.o',
        ]
    
    # Class operations
    
    def create_class(self, builder, module, class_name, base_classes=None, methods=None, attributes=None):
        """Create a new class."""
        return self.class_type.create_class(builder, module, class_name, base_classes, methods, attributes)
    
    def create_instance(self, builder, class_ptr):
        """Create a new instance of a class."""
        return self.class_type.create_instance(builder, class_ptr)
    
    def get_attribute(self, builder, module, instance_ptr, attr_name):
        """Get an attribute from an instance."""
        return self.class_type.get_attribute(builder, module, instance_ptr, attr_name)
    
    def set_attribute(self, builder, module, instance_ptr, attr_name, value):
        """Set an attribute on an instance."""
        self.class_type.set_attribute(builder, module, instance_ptr, attr_name, value)
    
    def call_method(self, builder, module, instance_ptr, method_name, args):
        """Call a method on an instance."""
        return self.class_type.call_method(builder, module, instance_ptr, method_name, args)
    
    # Inheritance operations
    
    def compute_mro(self, class_name, bases, all_classes):
        """Compute Method Resolution Order for a class."""
        return self.inheritance.compute_mro(class_name, bases, all_classes)
    
    def generate_super_call(self, builder, module, instance_ptr, current_class, method_name, args):
        """Generate super() method call."""
        return self.inheritance.generate_super_call(builder, module, instance_ptr, current_class, method_name, args)
    
    def check_isinstance(self, builder, module, instance_ptr, class_ptr):
        """Generate isinstance() check."""
        return self.inheritance.check_isinstance(builder, module, instance_ptr, class_ptr)
    
    def check_issubclass(self, builder, module, subclass_ptr, superclass_ptr):
        """Generate issubclass() check."""
        return self.inheritance.check_issubclass(builder, module, subclass_ptr, superclass_ptr)
    
    # Method dispatch operations
    
    def create_bound_method(self, builder, function_ptr, instance_ptr):
        """Create a bound method."""
        return self.method_dispatch.create_bound_method(builder, function_ptr, instance_ptr)
    
    def call_bound_method(self, builder, module, bound_method_ptr, args):
        """Call a bound method."""
        return self.method_dispatch.call_bound_method(builder, module, bound_method_ptr, args)
    
    def create_static_method(self, builder, function_ptr):
        """Create a static method."""
        return self.method_dispatch.create_static_method(builder, function_ptr)
    
    def create_class_method(self, builder, function_ptr, class_ptr):
        """Create a class method."""
        return self.method_dispatch.create_class_method(builder, function_ptr, class_ptr)
    
    def generate_vtable(self, builder, module, class_ptr, methods):
        """Generate virtual method table."""
        return self.method_dispatch.generate_vtable(builder, module, class_ptr, methods)
    
    def dynamic_dispatch(self, builder, module, instance_ptr, method_index):
        """Perform dynamic dispatch."""
        return self.method_dispatch.dynamic_dispatch(builder, module, instance_ptr, method_index)
    
    # Magic methods operations
    
    def generate_init(self, builder, module, instance_ptr, init_func, args):
        """Generate __init__ call."""
        self.magic_methods.generate_init(builder, module, instance_ptr, init_func, args)
    
    def generate_str(self, builder, module, instance_ptr):
        """Generate __str__ call."""
        return self.magic_methods.generate_str(builder, module, instance_ptr)
    
    def generate_repr(self, builder, module, instance_ptr):
        """Generate __repr__ call."""
        return self.magic_methods.generate_repr(builder, module, instance_ptr)
    
    def generate_eq(self, builder, module, left_ptr, right_ptr):
        """Generate __eq__ call."""
        return self.magic_methods.generate_eq(builder, module, left_ptr, right_ptr)
    
    def generate_hash(self, builder, module, instance_ptr):
        """Generate __hash__ call."""
        return self.magic_methods.generate_hash(builder, module, instance_ptr)
    
    def generate_len(self, builder, module, instance_ptr):
        """Generate __len__ call."""
        return self.magic_methods.generate_len(builder, module, instance_ptr)
    
    def generate_getitem(self, builder, module, instance_ptr, key):
        """Generate __getitem__ call."""
        return self.magic_methods.generate_getitem(builder, module, instance_ptr, key)
    
    def generate_setitem(self, builder, module, instance_ptr, key, value):
        """Generate __setitem__ call."""
        self.magic_methods.generate_setitem(builder, module, instance_ptr, key, value)
    
    def generate_iter(self, builder, module, instance_ptr):
        """Generate __iter__ call."""
        return self.magic_methods.generate_iter(builder, module, instance_ptr)
    
    def generate_next(self, builder, module, iterator_ptr):
        """Generate __next__ call."""
        return self.magic_methods.generate_next(builder, module, iterator_ptr)
    
    def generate_call(self, builder, module, instance_ptr, args):
        """Generate __call__ call."""
        return self.magic_methods.generate_call(builder, module, instance_ptr, args)
    
    def generate_binary_op(self, builder, module, op_name, left_ptr, right_ptr):
        """Generate binary operator magic method."""
        return self.magic_methods.generate_binary_op(builder, module, op_name, left_ptr, right_ptr)
    
    def generate_context_manager(self, builder, module, instance_ptr):
        """Generate __enter__ and __exit__ for context managers."""
        return self.magic_methods.generate_context_manager(builder, module, instance_ptr)
    
    # Property operations
    
    def create_property(self, builder, fget=None, fset=None, fdel=None, doc=None):
        """Create a property object."""
        return self.property_type.create_property(builder, fget, fset, fdel, doc)
    
    def property_get(self, builder, module, property_ptr, instance_ptr):
        """Call property getter."""
        return self.property_type.property_get(builder, module, property_ptr, instance_ptr)
    
    def property_set(self, builder, module, property_ptr, instance_ptr, value):
        """Call property setter."""
        self.property_type.property_set(builder, module, property_ptr, instance_ptr, value)
    
    def property_delete(self, builder, module, property_ptr, instance_ptr):
        """Call property deleter."""
        self.property_type.property_delete(builder, module, property_ptr, instance_ptr)
    
    def create_descriptor(self, builder, get_func, set_func, delete_func):
        """Create a descriptor object."""
        return self.property_type.create_descriptor(builder, get_func, set_func, delete_func)
    
    def descriptor_get(self, builder, module, descriptor_ptr, instance_ptr, owner_ptr):
        """Call descriptor __get__."""
        return self.property_type.descriptor_get(builder, module, descriptor_ptr, instance_ptr, owner_ptr)
    
    # Utility methods
    
    def get_runtime_objects(self):
        """Get list of runtime object files for linking."""
        return [obj for obj in self.runtime_objects if os.path.exists(obj)]
    
    def verify_runtime_compilation(self):
        """Verify all runtime object files are compiled."""
        missing = [obj for obj in self.runtime_objects if not os.path.exists(obj)]
        if missing:
            return False, missing
        return True, []
    
    def get_summary(self):
        """Get summary of Phase 3 OOP features."""
        return {
            'phase': 3,
            'name': 'Object-Oriented Programming',
            'components': {
                'classes': {
                    'class_creation': 'Full support',
                    'instance_creation': 'Full support',
                    'attributes': 'Get/set operations',
                    'methods': 'Method calls',
                },
                'inheritance': {
                    'single_inheritance': 'Full support',
                    'multiple_inheritance': 'Full support',
                    'mro': 'C3 linearization',
                    'super': 'super() calls',
                    'isinstance': 'Type checking',
                    'issubclass': 'Class hierarchy checking',
                },
                'method_dispatch': {
                    'bound_methods': 'Instance methods',
                    'static_methods': '@staticmethod',
                    'class_methods': '@classmethod',
                    'vtables': 'Virtual method tables',
                    'dynamic_dispatch': 'Runtime method lookup',
                },
                'magic_methods': {
                    'lifecycle': '__init__, __del__',
                    'representation': '__str__, __repr__',
                    'comparison': '__eq__, __ne__, __lt__, __le__, __gt__, __ge__',
                    'hashing': '__hash__',
                    'containers': '__len__, __getitem__, __setitem__, __delitem__',
                    'iteration': '__iter__, __next__',
                    'callable': '__call__',
                    'operators': '__add__, __sub__, __mul__, __truediv__, etc.',
                    'context_managers': '__enter__, __exit__',
                },
                'properties': {
                    'property': 'Property decorator',
                    'getter': '@property',
                    'setter': '@prop.setter',
                    'deleter': '@prop.deleter',
                    'descriptors': '__get__, __set__, __delete__',
                },
            },
            'runtime_objects': self.runtime_objects,
            'runtime_compiled': self.verify_runtime_compilation()[0],
        }


def demonstrate_phase3():
    """Demonstrate Phase 3 OOP capabilities."""
    
    print("=" * 60)
    print("Phase 3: Object-Oriented Programming Demonstration")
    print("=" * 60)
    
    oop = Phase3OOP()
    summary = oop.get_summary()
    
    print(f"\n📦 Phase {summary['phase']}: {summary['name']}")
    print("\n✅ Components:")
    
    for component, features in summary['components'].items():
        print(f"\n  {component.upper()}:")
        for feature, description in features.items():
            print(f"    • {feature}: {description}")
    
    print(f"\n🔧 Runtime Objects:")
    compiled, missing = oop.verify_runtime_compilation()
    for obj in oop.runtime_objects:
        status = "✅" if os.path.exists(obj) else "❌"
        size = ""
        if os.path.exists(obj):
            size_kb = os.path.getsize(obj) / 1024
            size = f" ({size_kb:.1f} KB)"
        print(f"  {status} {obj}{size}")
    
    if compiled:
        print("\n✅ All runtime objects compiled successfully!")
    else:
        print(f"\n⚠️  Missing runtime objects: {missing}")
    
    print("\n" + "=" * 60)
    print("Phase 3 Example Code Patterns")
    print("=" * 60)
    
    examples = [
        ("Basic Class", """
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name} ({self.age})"
    
    def greet(self):
        return f"Hello, I'm {self.name}"

p = Person("Alice", 30)
print(p)  # Alice (30)
print(p.greet())  # Hello, I'm Alice
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

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Rex")
print(dog.speak())  # Rex says Woof!
"""),
        ("Properties", """
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.area)  # 78.53975
c.radius = 10
print(c.area)  # 314.159
"""),
        ("Magic Methods", """
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

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2
print(v3)  # Vector(4, 6)
"""),
        ("Static and Class Methods", """
class MathUtils:
    pi = 3.14159
    
    @staticmethod
    def add(a, b):
        return a + b
    
    @classmethod
    def circle_area(cls, radius):
        return cls.pi * radius ** 2

print(MathUtils.add(5, 3))  # 8
print(MathUtils.circle_area(5))  # 78.53975
"""),
    ]
    
    for i, (title, code) in enumerate(examples, 1):
        print(f"\n{i}. {title}:")
        print(code)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_phase3()
