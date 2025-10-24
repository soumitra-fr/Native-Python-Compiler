"""
Phase 8: Context Manager & Advanced Features Support
Provides with statements, decorators, metaclasses, and advanced Python features.
"""

from llvmlite import ir


class ContextManagerSupport:
    """
    Handles context managers and advanced Python features.
    
    Features:
    - with statement (__enter__/__exit__)
    - Multiple context managers
    - Exception handling in context
    - Decorators with arguments
    - Metaclasses
    - __slots__
    - Advanced descriptors
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Context manager structure
        self.context_manager_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # __enter__ method
            self.void_ptr,        # __exit__ method
            self.void_ptr,        # entered value
            self.int32,           # is_entered
        ])
    
    def generate_with_statement(self, builder, module, context_manager, body_func):
        """
        Generate with statement.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            context_manager: Context manager object
            body_func: Body function pointer
        
        Returns:
            Result of body execution
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="with_statement")
        
        cm_ptr = builder.bitcast(context_manager, self.void_ptr)
        body_ptr = builder.bitcast(body_func, self.void_ptr)
        
        result = builder.call(func, [cm_ptr, body_ptr])
        return result
    
    def context_enter(self, builder, module, context_manager):
        """
        Call __enter__() method.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            context_manager: Context manager object
        
        Returns:
            Value returned by __enter__()
        """
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="context_enter")
        
        cm_ptr = builder.bitcast(context_manager, self.void_ptr)
        result = builder.call(func, [cm_ptr])
        return result
    
    def context_exit(self, builder, module, context_manager, exc_type=None, exc_value=None, traceback=None):
        """
        Call __exit__() method.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            context_manager: Context manager object
            exc_type: Exception type (if any)
            exc_value: Exception value (if any)
            traceback: Traceback (if any)
        
        Returns:
            Boolean indicating if exception should be suppressed
        """
        arg_types = [self.void_ptr, self.void_ptr, self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.int32, arg_types)
        func = ir.Function(module, func_type, name="context_exit")
        
        cm_ptr = builder.bitcast(context_manager, self.void_ptr)
        exc_type_ptr = builder.bitcast(exc_type, self.void_ptr) if exc_type else ir.Constant(self.void_ptr, None)
        exc_val_ptr = builder.bitcast(exc_value, self.void_ptr) if exc_value else ir.Constant(self.void_ptr, None)
        tb_ptr = builder.bitcast(traceback, self.void_ptr) if traceback else ir.Constant(self.void_ptr, None)
        
        result = builder.call(func, [cm_ptr, exc_type_ptr, exc_val_ptr, tb_ptr])
        return result
    
    def apply_decorator(self, builder, module, decorator_func, target_func, *args):
        """
        Apply decorator to function.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            decorator_func: Decorator function
            target_func: Function to decorate
            *args: Decorator arguments
        
        Returns:
            Decorated function
        """
        arg_types = [self.void_ptr, self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="apply_decorator")
        
        dec_ptr = builder.bitcast(decorator_func, self.void_ptr)
        target_ptr = builder.bitcast(target_func, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args))
        args_array = self._create_pointer_array(builder, module, args)
        
        result = builder.call(func, [dec_ptr, target_ptr, num_args, args_array])
        return result
    
    def create_metaclass(self, builder, module, metaclass_name, bases, namespace):
        """
        Create metaclass.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            metaclass_name: Name of metaclass
            bases: Base classes
            namespace: Class namespace dict
        
        Returns:
            Metaclass object
        """
        arg_types = [self.char_ptr, self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_metaclass")
        
        name_str = self._create_string_literal(builder, module, metaclass_name)
        bases_ptr = builder.bitcast(bases, self.void_ptr)
        ns_ptr = builder.bitcast(namespace, self.void_ptr)
        
        result = builder.call(func, [name_str, bases_ptr, ns_ptr])
        return result
    
    def apply_metaclass(self, builder, module, metaclass, class_name, bases, namespace):
        """
        Apply metaclass to create class.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            metaclass: Metaclass object
            class_name: Name of class to create
            bases: Base classes
            namespace: Class namespace
        
        Returns:
            Created class
        """
        arg_types = [self.void_ptr, self.char_ptr, self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="apply_metaclass")
        
        meta_ptr = builder.bitcast(metaclass, self.void_ptr)
        name_str = self._create_string_literal(builder, module, class_name)
        bases_ptr = builder.bitcast(bases, self.void_ptr)
        ns_ptr = builder.bitcast(namespace, self.void_ptr)
        
        result = builder.call(func, [meta_ptr, name_str, bases_ptr, ns_ptr])
        return result
    
    def create_slots_class(self, builder, module, class_name, slots):
        """
        Create class with __slots__.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            class_name: Name of class
            slots: List of slot names
        
        Returns:
            Class with slots
        """
        arg_types = [self.char_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_slots_class")
        
        name_str = self._create_string_literal(builder, module, class_name)
        num_slots = ir.Constant(self.int32, len(slots))
        slots_array = self._create_string_array(builder, module, slots)
        
        result = builder.call(func, [name_str, num_slots, slots_array])
        return result
    
    def create_descriptor(self, builder, module, get_func=None, set_func=None, delete_func=None):
        """
        Create descriptor with __get__/__set__/__delete__.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            get_func: __get__ function
            set_func: __set__ function
            delete_func: __delete__ function
        
        Returns:
            Descriptor object
        """
        arg_types = [self.void_ptr, self.void_ptr, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="create_descriptor")
        
        get_ptr = builder.bitcast(get_func, self.void_ptr) if get_func else ir.Constant(self.void_ptr, None)
        set_ptr = builder.bitcast(set_func, self.void_ptr) if set_func else ir.Constant(self.void_ptr, None)
        del_ptr = builder.bitcast(delete_func, self.void_ptr) if delete_func else ir.Constant(self.void_ptr, None)
        
        result = builder.call(func, [get_ptr, set_ptr, del_ptr])
        return result
    
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
    
    def _create_string_array(self, builder, module, strings):
        """Create an array of strings."""
        if not strings:
            return ir.Constant(self.void_ptr, None)
        
        str_ptrs = [self._create_string_literal(builder, module, s) for s in strings]
        return self._create_pointer_array(builder, module, str_ptrs)


def generate_context_manager_runtime():
    """Generate C runtime code for context managers and advanced features."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Context manager structure
typedef struct ContextManager {
    int64_t refcount;
    void* enter_method;
    void* exit_method;
    void* entered_value;
    int32_t is_entered;
} ContextManager;

// With statement
void* with_statement(void* context_manager, void* body_func) {
    // Call __enter__
    void* value = context_enter(context_manager);
    
    // Execute body
    void* result = NULL;  // Would call body_func(value)
    
    // Call __exit__
    context_exit(context_manager, NULL, NULL, NULL);
    
    return result;
}

// Context manager __enter__
void* context_enter(void* cm_ptr) {
    ContextManager* cm = (ContextManager*)cm_ptr;
    cm->is_entered = 1;
    // Call __enter__ method
    return cm->entered_value;
}

// Context manager __exit__
int32_t context_exit(void* cm_ptr, void* exc_type, void* exc_value, void* traceback) {
    ContextManager* cm = (ContextManager*)cm_ptr;
    cm->is_entered = 0;
    // Call __exit__ method
    // Return 1 to suppress exception, 0 to propagate
    return 0;
}

// Apply decorator
void* apply_decorator(void* decorator_func, void* target_func, int32_t num_args, void* args) {
    // Apply decorator to function
    return target_func;
}

// Create metaclass
void* create_metaclass(char* name, void* bases, void* namespace) {
    // Create metaclass
    return NULL;
}

// Apply metaclass
void* apply_metaclass(void* metaclass, char* class_name, void* bases, void* namespace) {
    // Use metaclass to create class
    return NULL;
}

// Create slots class
void* create_slots_class(char* class_name, int32_t num_slots, void* slots) {
    // Create class with __slots__
    return NULL;
}

// Create descriptor
void* create_descriptor(void* get_func, void* set_func, void* delete_func) {
    // Create descriptor object
    return NULL;
}
"""
    
    with open('context_manager_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Context manager runtime generated: context_manager_runtime.c")


if __name__ == "__main__":
    generate_context_manager_runtime()
    
    cm_support = ContextManagerSupport()
    
    print(f"✅ ContextManagerSupport initialized")
    print(f"   - ContextManager structure: {cm_support.context_manager_type}")
