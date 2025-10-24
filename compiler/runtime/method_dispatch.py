"""
Phase 3: Method Dispatch Implementation
Provides virtual method tables, dynamic dispatch, and method types.
"""

from llvmlite import ir

class MethodDispatch:
    """
    Handles method dispatch including:
    - Virtual method tables (vtables)
    - Dynamic dispatch
    - Bound and unbound methods
    - staticmethod and classmethod
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.int1 = ir.IntType(1)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # BoundMethod structure
        self.bound_method_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # function pointer
            self.void_ptr,        # self (instance)
        ])
        
        # StaticMethod structure
        self.static_method_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # function pointer
        ])
        
        # ClassMethod structure
        self.class_method_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # function pointer
            self.void_ptr,        # class pointer
        ])
    
    def create_bound_method(self, builder, function_ptr, instance_ptr):
        """
        Create a bound method (method bound to an instance).
        
        Args:
            builder: LLVM IR builder
            function_ptr: Pointer to function
            instance_ptr: Pointer to instance (self)
        
        Returns:
            Pointer to BoundMethod structure
        """
        # Allocate bound method structure
        method_ptr = builder.alloca(self.bound_method_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set function pointer
        func_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        func_void_ptr = builder.bitcast(function_ptr, self.void_ptr)
        builder.store(func_void_ptr, func_ptr)
        
        # Set self (instance)
        self_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        builder.store(instance_void_ptr, self_ptr)
        
        return method_ptr
    
    def call_bound_method(self, builder, module, bound_method_ptr, args):
        """
        Call a bound method (automatically passes self as first arg).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            bound_method_ptr: Pointer to BoundMethod
            args: List of additional arguments
        
        Returns:
            Return value from method
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="call_bound_method")
        
        # Create args array
        args_array = self._create_pointer_array(builder, module, args) if args else ir.Constant(self.void_ptr, None)
        
        # Call runtime function
        method_void_ptr = builder.bitcast(bound_method_ptr, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args) if args else 0)
        
        result = builder.call(func, [method_void_ptr, num_args, args_array])
        
        return result
    
    def create_static_method(self, builder, function_ptr):
        """
        Create a static method (no self or cls).
        
        Args:
            builder: LLVM IR builder
            function_ptr: Pointer to function
        
        Returns:
            Pointer to StaticMethod structure
        """
        # Allocate static method structure
        method_ptr = builder.alloca(self.static_method_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set function pointer
        func_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        func_void_ptr = builder.bitcast(function_ptr, self.void_ptr)
        builder.store(func_void_ptr, func_ptr)
        
        return method_ptr
    
    def create_class_method(self, builder, function_ptr, class_ptr):
        """
        Create a class method (receives class as first arg).
        
        Args:
            builder: LLVM IR builder
            function_ptr: Pointer to function
            class_ptr: Pointer to class (cls)
        
        Returns:
            Pointer to ClassMethod structure
        """
        # Allocate class method structure
        method_ptr = builder.alloca(self.class_method_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set function pointer
        func_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        func_void_ptr = builder.bitcast(function_ptr, self.void_ptr)
        builder.store(func_void_ptr, func_ptr)
        
        # Set cls (class)
        cls_ptr = builder.gep(method_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        class_void_ptr = builder.bitcast(class_ptr, self.void_ptr)
        builder.store(class_void_ptr, cls_ptr)
        
        return method_ptr
    
    def generate_vtable(self, builder, module, class_ptr, methods):
        """
        Generate virtual method table for a class.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_ptr: Pointer to Class
            methods: Dict of method_name -> function_ptr
        
        Returns:
            Pointer to vtable
        """
        # Create vtable as array of function pointers
        if not methods:
            return ir.Constant(self.void_ptr, None)
        
        vtable_size = len(methods)
        vtable_type = ir.ArrayType(self.void_ptr, vtable_size)
        vtable_ptr = builder.alloca(vtable_type)
        
        # Fill vtable
        for i, (name, func_ptr) in enumerate(methods.items()):
            slot_ptr = builder.gep(vtable_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            func_void_ptr = builder.bitcast(func_ptr, self.void_ptr)
            builder.store(func_void_ptr, slot_ptr)
        
        return builder.bitcast(vtable_ptr, self.void_ptr)
    
    def dynamic_dispatch(self, builder, module, instance_ptr, method_index):
        """
        Perform dynamic dispatch using vtable.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            method_index: Index in vtable
        
        Returns:
            Function pointer from vtable
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.int32])
        func = ir.Function(module, func_type, name="vtable_lookup")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        index = ir.Constant(self.int32, method_index)
        
        result = builder.call(func, [instance_void_ptr, index])
        
        return result
    
    def get_method_from_name(self, builder, module, class_ptr, method_name):
        """
        Get method function pointer from name.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_ptr: Pointer to Class
            method_name: Name of method
        
        Returns:
            Function pointer
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="get_method_by_name")
        
        name_str = self._create_string_literal(builder, module, method_name)
        class_void_ptr = builder.bitcast(class_ptr, self.void_ptr)
        
        result = builder.call(func, [class_void_ptr, name_str])
        
        return result
    
    def incref(self, builder, module, method_ptr):
        """Increment reference count."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="method_incref")
        method_void_ptr = builder.bitcast(method_ptr, self.void_ptr)
        builder.call(func, [method_void_ptr])
    
    def decref(self, builder, module, method_ptr):
        """Decrement reference count and free if zero."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="method_decref")
        method_void_ptr = builder.bitcast(method_ptr, self.void_ptr)
        builder.call(func, [method_void_ptr])
    
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
        
        array_type = ir.ArrayType(self.void_ptr, len(pointers))
        array_ptr = builder.alloca(array_type)
        
        for i, ptr in enumerate(pointers):
            elem_ptr = builder.gep(array_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            ptr_void = builder.bitcast(ptr, self.void_ptr)
            builder.store(ptr_void, elem_ptr)
        
        return builder.bitcast(array_ptr, self.void_ptr)


def generate_method_dispatch_runtime():
    """Generate C runtime code for method dispatch."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Forward declarations
typedef struct Class Class;
typedef struct Instance Instance;

struct Class {
    int64_t refcount;
    char* name;
    Class** bases;
    int32_t num_bases;
    void** methods;
    int32_t num_methods;
    char** method_names;
    void** attributes;
    char** attr_names;
    int32_t num_attrs;
};

struct Instance {
    int64_t refcount;
    Class* cls;
    void** attributes;
    char** attr_names;
    int32_t num_attrs;
};

// BoundMethod structure
typedef struct BoundMethod {
    int64_t refcount;
    void* function;
    void* self;
} BoundMethod;

// StaticMethod structure
typedef struct StaticMethod {
    int64_t refcount;
    void* function;
} StaticMethod;

// ClassMethod structure
typedef struct ClassMethod {
    int64_t refcount;
    void* function;
    void* cls;
} ClassMethod;

// Reference counting
void method_incref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)++;
    }
}

void method_decref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)--;
        if (*refcount == 0) {
            free(obj);
        }
    }
}

// Call bound method
void* call_bound_method(void* bound_method_ptr, int32_t num_args, void* args_array) {
    BoundMethod* method = (BoundMethod*)bound_method_ptr;
    
    // Would call function with self as first arg
    // For now, return NULL
    return NULL;
}

// Vtable lookup
void* vtable_lookup(void* instance_ptr, int32_t method_index) {
    Instance* inst = (Instance*)instance_ptr;
    Class* cls = inst->cls;
    
    if (method_index < cls->num_methods) {
        return cls->methods[method_index];
    }
    
    return NULL;
}

// Get method by name
void* get_method_by_name(void* class_ptr, char* method_name) {
    Class* cls = (Class*)class_ptr;
    
    for (int32_t i = 0; i < cls->num_methods; i++) {
        if (strcmp(cls->method_names[i], method_name) == 0) {
            return cls->methods[i];
        }
    }
    
    return NULL;
}
"""
    
    # Write to file
    with open('method_dispatch_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Method dispatch runtime generated: method_dispatch_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_method_dispatch_runtime()
    
    # Test method dispatch structures
    dispatch = MethodDispatch()
    print(f"✅ MethodDispatch initialized")
    print(f"   - BoundMethod structure: {dispatch.bound_method_type}")
    print(f"   - StaticMethod structure: {dispatch.static_method_type}")
    print(f"   - ClassMethod structure: {dispatch.class_method_type}")
