"""
Phase 3: Class Type Implementation
Provides basic class structure with instances, attributes, and methods.
"""

from llvmlite import ir
import ctypes

class ClassType:
    """
    Represents a Python class type in LLVM IR.
    
    Memory Layout:
    struct Class {
        int64_t refcount;
        char* name;
        struct Class** bases;      // Array of base classes
        int32_t num_bases;
        void** methods;            // Method table (vtable)
        int32_t num_methods;
        char** method_names;
        void** attributes;         // Class attributes
        char** attr_names;
        int32_t num_attrs;
    }
    
    struct Instance {
        int64_t refcount;
        struct Class* cls;         // Pointer to class
        void** attributes;         // Instance attributes
        char** attr_names;
        int32_t num_attrs;
    }
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Class structure
        self.class_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.char_ptr,        # name
            self.void_ptr,        # bases (Class**)
            self.int32,           # num_bases
            self.void_ptr,        # methods (void**)
            self.int32,           # num_methods
            self.void_ptr,        # method_names (char**)
            self.void_ptr,        # attributes (void**)
            self.void_ptr,        # attr_names (char**)
            self.int32,           # num_attrs
        ])
        
        # Instance structure
        self.instance_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # cls (Class*)
            self.void_ptr,        # attributes (void**)
            self.void_ptr,        # attr_names (char**)
            self.int32,           # num_attrs
        ])
        
    def create_class(self, builder, module, class_name, base_classes=None, methods=None, attributes=None):
        """
        Create a new class object.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_name: Name of the class (string)
            base_classes: List of base class pointers (for inheritance)
            methods: Dict of method_name -> function pointer
            attributes: Dict of attr_name -> initial value
        
        Returns:
            Pointer to Class structure
        """
        # Allocate class structure
        class_ptr = builder.alloca(self.class_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set class name
        name_str = self._create_string_literal(builder, module, class_name)
        name_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        builder.store(name_str, name_ptr)
        
        # Set base classes
        if base_classes:
            bases_array = self._create_pointer_array(builder, module, base_classes)
            bases_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
            builder.store(bases_array, bases_ptr)
            
            num_bases_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
            builder.store(ir.Constant(self.int32, len(base_classes)), num_bases_ptr)
        else:
            bases_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
            builder.store(ir.Constant(self.void_ptr, None), bases_ptr)
            num_bases_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
            builder.store(ir.Constant(self.int32, 0), num_bases_ptr)
        
        # Set methods
        if methods:
            method_ptrs, method_names = self._create_method_table(builder, module, methods)
            methods_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
            builder.store(method_ptrs, methods_ptr)
            
            num_methods_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 5)])
            builder.store(ir.Constant(self.int32, len(methods)), num_methods_ptr)
            
            method_names_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 6)])
            builder.store(method_names, method_names_ptr)
        else:
            methods_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
            builder.store(ir.Constant(self.void_ptr, None), methods_ptr)
            num_methods_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 5)])
            builder.store(ir.Constant(self.int32, 0), num_methods_ptr)
            method_names_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 6)])
            builder.store(ir.Constant(self.void_ptr, None), method_names_ptr)
        
        # Set class attributes
        if attributes:
            attr_ptrs, attr_names = self._create_attribute_table(builder, module, attributes)
            attrs_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 7)])
            builder.store(attr_ptrs, attrs_ptr)
            
            attr_names_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 8)])
            builder.store(attr_names, attr_names_ptr)
            
            num_attrs_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 9)])
            builder.store(ir.Constant(self.int32, len(attributes)), num_attrs_ptr)
        else:
            attrs_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 7)])
            builder.store(ir.Constant(self.void_ptr, None), attrs_ptr)
            attr_names_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 8)])
            builder.store(ir.Constant(self.void_ptr, None), attr_names_ptr)
            num_attrs_ptr = builder.gep(class_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 9)])
            builder.store(ir.Constant(self.int32, 0), num_attrs_ptr)
        
        return class_ptr
    
    def create_instance(self, builder, class_ptr):
        """
        Create a new instance of a class.
        
        Args:
            builder: LLVM IR builder
            class_ptr: Pointer to Class structure
        
        Returns:
            Pointer to Instance structure
        """
        # Allocate instance structure
        instance_ptr = builder.alloca(self.instance_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(instance_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set class pointer
        cls_ptr = builder.gep(instance_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        class_void_ptr = builder.bitcast(class_ptr, self.void_ptr)
        builder.store(class_void_ptr, cls_ptr)
        
        # Initialize empty attributes
        attrs_ptr = builder.gep(instance_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        builder.store(ir.Constant(self.void_ptr, None), attrs_ptr)
        
        attr_names_ptr = builder.gep(instance_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
        builder.store(ir.Constant(self.void_ptr, None), attr_names_ptr)
        
        num_attrs_ptr = builder.gep(instance_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
        builder.store(ir.Constant(self.int32, 0), num_attrs_ptr)
        
        return instance_ptr
    
    def get_attribute(self, builder, module, instance_ptr, attr_name):
        """
        Get an attribute from an instance.
        Uses runtime lookup function.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance structure
            attr_name: Name of attribute to get
        
        Returns:
            Value of attribute (void*)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="instance_get_attr")
        
        # Create attribute name string
        name_str = self._create_string_literal(builder, module, attr_name)
        
        # Call runtime function
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        result = builder.call(func, [instance_void_ptr, name_str])
        
        return result
    
    def set_attribute(self, builder, module, instance_ptr, attr_name, value):
        """
        Set an attribute on an instance.
        Uses runtime function.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance structure
            attr_name: Name of attribute to set
            value: Value to set (void*)
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.char_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="instance_set_attr")
        
        # Create attribute name string
        name_str = self._create_string_literal(builder, module, attr_name)
        
        # Call runtime function
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        value_void_ptr = builder.bitcast(value, self.void_ptr)
        builder.call(func, [instance_void_ptr, name_str, value_void_ptr])
    
    def call_method(self, builder, module, instance_ptr, method_name, args):
        """
        Call a method on an instance.
        Looks up method in class and calls with self as first argument.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance structure
            method_name: Name of method to call
            args: List of argument values
        
        Returns:
            Return value from method
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.char_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="instance_call_method")
        
        # Create method name string
        name_str = self._create_string_literal(builder, module, method_name)
        
        # Create args array
        args_array = self._create_pointer_array(builder, module, args) if args else ir.Constant(self.void_ptr, None)
        
        # Call runtime function
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args) if args else 0)
        result = builder.call(func, [instance_void_ptr, name_str, num_args, args_array])
        
        return result
    
    def incref(self, builder, module, obj_ptr):
        """Increment reference count."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="class_incref")
        obj_void_ptr = builder.bitcast(obj_ptr, self.void_ptr)
        builder.call(func, [obj_void_ptr])
    
    def decref(self, builder, module, obj_ptr):
        """Decrement reference count and free if zero."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="class_decref")
        obj_void_ptr = builder.bitcast(obj_ptr, self.void_ptr)
        builder.call(func, [obj_void_ptr])
    
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
    
    def _create_pointer_array(self, builder, module, pointers):
        """Create an array of void pointers."""
        if not pointers:
            return ir.Constant(self.void_ptr, None)
        
        # Allocate array
        array_type = ir.ArrayType(self.void_ptr, len(pointers))
        array_ptr = builder.alloca(array_type)
        
        # Fill array
        for i, ptr in enumerate(pointers):
            elem_ptr = builder.gep(array_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            ptr_void = builder.bitcast(ptr, self.void_ptr)
            builder.store(ptr_void, elem_ptr)
        
        return builder.bitcast(array_ptr, self.void_ptr)
    
    def _create_method_table(self, builder, module, methods):
        """Create method table (function pointers and names)."""
        method_ptrs = []
        method_names = []
        
        for name, func_ptr in methods.items():
            method_ptrs.append(func_ptr)
            method_names.append(self._create_string_literal(builder, module, name))
        
        ptrs_array = self._create_pointer_array(builder, module, method_ptrs)
        names_array = self._create_pointer_array(builder, module, method_names)
        
        return ptrs_array, names_array
    
    def _create_attribute_table(self, builder, module, attributes):
        """Create attribute table (values and names)."""
        attr_ptrs = []
        attr_names = []
        
        for name, value in attributes.items():
            attr_ptrs.append(value)
            attr_names.append(self._create_string_literal(builder, module, name))
        
        ptrs_array = self._create_pointer_array(builder, module, attr_ptrs)
        names_array = self._create_pointer_array(builder, module, attr_names)
        
        return ptrs_array, names_array


def generate_class_runtime():
    """Generate C runtime code for class operations."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Class structure
typedef struct Class {
    int64_t refcount;
    char* name;
    struct Class** bases;
    int32_t num_bases;
    void** methods;
    int32_t num_methods;
    char** method_names;
    void** attributes;
    char** attr_names;
    int32_t num_attrs;
} Class;

// Instance structure
typedef struct Instance {
    int64_t refcount;
    Class* cls;
    void** attributes;
    char** attr_names;
    int32_t num_attrs;
} Instance;

// Reference counting
void class_incref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)++;
    }
}

void class_decref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)--;
        if (*refcount == 0) {
            free(obj);
        }
    }
}

// Instance attribute access
void* instance_get_attr(void* instance_ptr, char* attr_name) {
    Instance* inst = (Instance*)instance_ptr;
    
    // Search in instance attributes
    for (int32_t i = 0; i < inst->num_attrs; i++) {
        if (strcmp(inst->attr_names[i], attr_name) == 0) {
            return inst->attributes[i];
        }
    }
    
    // Search in class attributes
    Class* cls = inst->cls;
    for (int32_t i = 0; i < cls->num_attrs; i++) {
        if (strcmp(cls->attr_names[i], attr_name) == 0) {
            return cls->attributes[i];
        }
    }
    
    return NULL;  // AttributeError
}

void instance_set_attr(void* instance_ptr, char* attr_name, void* value) {
    Instance* inst = (Instance*)instance_ptr;
    
    // Check if attribute exists
    for (int32_t i = 0; i < inst->num_attrs; i++) {
        if (strcmp(inst->attr_names[i], attr_name) == 0) {
            inst->attributes[i] = value;
            return;
        }
    }
    
    // Add new attribute
    inst->num_attrs++;
    inst->attributes = realloc(inst->attributes, inst->num_attrs * sizeof(void*));
    inst->attr_names = realloc(inst->attr_names, inst->num_attrs * sizeof(char*));
    inst->attributes[inst->num_attrs - 1] = value;
    inst->attr_names[inst->num_attrs - 1] = strdup(attr_name);
}

// Method call (simplified - actual dispatch in method_dispatch.c)
void* instance_call_method(void* instance_ptr, char* method_name, int32_t num_args, void* args_array) {
    Instance* inst = (Instance*)instance_ptr;
    Class* cls = inst->cls;
    
    // Find method in class
    for (int32_t i = 0; i < cls->num_methods; i++) {
        if (strcmp(cls->method_names[i], method_name) == 0) {
            // Method found - would call here
            // For now, return NULL
            return NULL;
        }
    }
    
    return NULL;  // AttributeError
}
"""
    
    # Write to file
    with open('class_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Class runtime generated: class_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_class_runtime()
    
    # Test class type structure
    class_type = ClassType()
    print(f"✅ ClassType initialized")
    print(f"   - Class structure: {class_type.class_type}")
    print(f"   - Instance structure: {class_type.instance_type}")
