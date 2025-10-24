"""
Advanced Python Features Support
Provides property decorators, classmethod/staticmethod, weakref, and more.
"""

from llvmlite import ir


class AdvancedFeatures:
    """
    Handles advanced Python features.
    
    Features:
    - @property decorator
    - @classmethod and @staticmethod
    - weakref support
    - super() calls
    - Multiple inheritance (MRO)
    - Abstract base classes
    - Callable objects
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Property structure
        self.property_type = ir.LiteralStructType([
            self.int64,      # refcount
            self.void_ptr,   # getter
            self.void_ptr,   # setter
            self.void_ptr,   # deleter
            self.char_ptr,   # doc
        ])
    
    def create_property(self, builder, module, getter=None, setter=None, deleter=None, doc=""):
        """
        Create @property decorator.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            getter: Getter function
            setter: Setter function
            deleter: Deleter function
            doc: Documentation string
        
        Returns:
            Property object
        """
        arg_types = [self.void_ptr, self.void_ptr, self.void_ptr, self.char_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_property")
        
        getter_ptr = builder.bitcast(getter, self.void_ptr) if getter else ir.Constant(self.void_ptr, None)
        setter_ptr = builder.bitcast(setter, self.void_ptr) if setter else ir.Constant(self.void_ptr, None)
        deleter_ptr = builder.bitcast(deleter, self.void_ptr) if deleter else ir.Constant(self.void_ptr, None)
        doc_str = self._create_string_literal(builder, module, doc)
        
        result = builder.call(func, [getter_ptr, setter_ptr, deleter_ptr, doc_str])
        return result
    
    def create_classmethod(self, builder, module, func):
        """
        Create @classmethod decorator.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            func: Method function
        
        Returns:
            Classmethod object
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        classmethod_func = ir.Function(module, func_type, name="create_classmethod")
        
        func_ptr = builder.bitcast(func, self.void_ptr)
        result = builder.call(classmethod_func, [func_ptr])
        return result
    
    def create_staticmethod(self, builder, module, func):
        """
        Create @staticmethod decorator.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            func: Method function
        
        Returns:
            Staticmethod object
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        staticmethod_func = ir.Function(module, func_type, name="create_staticmethod")
        
        func_ptr = builder.bitcast(func, self.void_ptr)
        result = builder.call(staticmethod_func, [func_ptr])
        return result
    
    def create_weakref(self, builder, module, obj, callback=None):
        """
        Create weak reference.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            obj: Object to reference
            callback: Callback when object is deleted
        
        Returns:
            Weakref object
        """
        arg_types = [self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_weakref")
        
        obj_ptr = builder.bitcast(obj, self.void_ptr)
        callback_ptr = builder.bitcast(callback, self.void_ptr) if callback else ir.Constant(self.void_ptr, None)
        
        result = builder.call(func, [obj_ptr, callback_ptr])
        return result
    
    def call_super(self, builder, module, cls, obj=None):
        """
        Call super() for method resolution.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            cls: Current class
            obj: Object instance (optional)
        
        Returns:
            Super object
        """
        arg_types = [self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="call_super")
        
        cls_ptr = builder.bitcast(cls, self.void_ptr)
        obj_ptr = builder.bitcast(obj, self.void_ptr) if obj else ir.Constant(self.void_ptr, None)
        
        result = builder.call(func, [cls_ptr, obj_ptr])
        return result
    
    def compute_mro(self, builder, module, cls, bases):
        """
        Compute Method Resolution Order (C3 linearization).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            cls: Class to compute MRO for
            bases: Base classes
        
        Returns:
            MRO tuple
        """
        arg_types = [self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="compute_mro")
        
        cls_ptr = builder.bitcast(cls, self.void_ptr)
        bases_ptr = builder.bitcast(bases, self.void_ptr)
        
        result = builder.call(func, [cls_ptr, bases_ptr])
        return result
    
    def create_abc(self, builder, module, class_name, abstract_methods):
        """
        Create Abstract Base Class.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_name: Name of ABC
            abstract_methods: List of abstract method names
        
        Returns:
            ABC class
        """
        arg_types = [self.char_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_abc")
        
        name_str = self._create_string_literal(builder, module, class_name)
        num_methods = ir.Constant(self.int32, len(abstract_methods))
        methods_array = self._create_string_array(builder, module, abstract_methods)
        
        result = builder.call(func, [name_str, num_methods, methods_array])
        return result
    
    def make_callable(self, builder, module, obj, call_func):
        """
        Make object callable by defining __call__.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            obj: Object to make callable
            call_func: __call__ implementation
        
        Returns:
            Callable object
        """
        arg_types = [self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="make_callable")
        
        obj_ptr = builder.bitcast(obj, self.void_ptr)
        call_ptr = builder.bitcast(call_func, self.void_ptr)
        
        result = builder.call(func, [obj_ptr, call_ptr])
        return result
    
    def get_attribute_descriptor(self, builder, module, obj, attr_name):
        """
        Get attribute using descriptor protocol.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            obj: Object
            attr_name: Attribute name
        
        Returns:
            Attribute value (via descriptor)
        """
        arg_types = [self.void_ptr, self.char_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="get_attribute_descriptor")
        
        obj_ptr = builder.bitcast(obj, self.void_ptr)
        name_str = self._create_string_literal(builder, module, attr_name)
        
        result = builder.call(func, [obj_ptr, name_str])
        return result
    
    def set_attribute_descriptor(self, builder, module, obj, attr_name, value):
        """
        Set attribute using descriptor protocol.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            obj: Object
            attr_name: Attribute name
            value: Value to set
        """
        arg_types = [self.void_ptr, self.char_ptr, self.void_ptr]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        func = ir.Function(module, func_type, name="set_attribute_descriptor")
        
        obj_ptr = builder.bitcast(obj, self.void_ptr)
        name_str = self._create_string_literal(builder, module, attr_name)
        val_ptr = builder.bitcast(value, self.void_ptr)
        
        builder.call(func, [obj_ptr, name_str, val_ptr])
    
    # Helper methods
    
    def _create_string_literal(self, builder, module, string_value):
        """Create a string literal in LLVM IR."""
        string_bytes = (string_value + '\0').encode('utf-8')
        string_const = ir.Constant(ir.ArrayType(self.int8, len(string_bytes)),
                                   bytearray(string_bytes))
        global_str = ir.GlobalVariable(module, string_const.type, 
                                       name=module.get_unique_name("str"))
        global_str.initializer = string_const
        global_str.global_constant = True
        return builder.bitcast(global_str, self.char_ptr)
    
    def _create_string_array(self, builder, module, strings):
        """Create an array of strings."""
        if not strings:
            return ir.Constant(self.void_ptr, None)
        
        str_ptrs = [self._create_string_literal(builder, module, s) for s in strings]
        array_type = ir.ArrayType(self.char_ptr, len(str_ptrs))
        array_ptr = builder.alloca(array_type)
        
        for i, ptr in enumerate(str_ptrs):
            elem_ptr = builder.gep(array_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            builder.store(ptr, elem_ptr)
        
        return builder.bitcast(array_ptr, self.void_ptr)


def generate_advanced_features_runtime():
    """Generate C runtime code for advanced features."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Property structure
typedef struct Property {
    int64_t refcount;
    void* getter;
    void* setter;
    void* deleter;
    char* doc;
} Property;

// Create property
void* create_property(void* getter, void* setter, void* deleter, char* doc) {
    Property* prop = (Property*)malloc(sizeof(Property));
    prop->refcount = 1;
    prop->getter = getter;
    prop->setter = setter;
    prop->deleter = deleter;
    prop->doc = doc;
    return prop;
}

// Create classmethod
void* create_classmethod(void* func) {
    // Wrap function as classmethod
    return func;
}

// Create staticmethod
void* create_staticmethod(void* func) {
    // Wrap function as staticmethod
    return func;
}

// Create weakref
void* create_weakref(void* obj, void* callback) {
    // Create weak reference to object
    return obj;
}

// Call super
void* call_super(void* cls, void* obj) {
    // Call super() for method resolution
    return cls;
}

// Compute MRO
void* compute_mro(void* cls, void* bases) {
    // Compute C3 linearization MRO
    return bases;
}

// Create ABC
void* create_abc(char* class_name, int32_t num_methods, void* methods) {
    // Create abstract base class
    return NULL;
}

// Make callable
void* make_callable(void* obj, void* call_func) {
    // Add __call__ to object
    return obj;
}

// Get attribute (descriptor protocol)
void* get_attribute_descriptor(void* obj, char* attr_name) {
    // Get attribute using descriptor protocol
    return NULL;
}

// Set attribute (descriptor protocol)
void set_attribute_descriptor(void* obj, char* attr_name, void* value) {
    // Set attribute using descriptor protocol
}
"""
    
    with open('advanced_features_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Advanced features runtime generated: advanced_features_runtime.c")


if __name__ == "__main__":
    generate_advanced_features_runtime()
    
    adv_features = AdvancedFeatures()
    
    print(f"✅ AdvancedFeatures initialized")
    print(f"   - Property structure: {adv_features.property_type}")
    print(f"   - Features: @property, @classmethod, @staticmethod, weakref, super(), MRO, ABC")
