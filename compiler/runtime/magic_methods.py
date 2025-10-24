"""
Phase 3: Magic Methods Implementation
Provides support for Python's special/magic methods (dunder methods).
"""

from llvmlite import ir

class MagicMethods:
    """
    Handles Python magic methods (__init__, __str__, __repr__, etc.).
    
    Magic methods enable operator overloading and special behaviors.
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.int1 = ir.IntType(1)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Supported magic methods
        self.magic_method_names = [
            '__init__', '__del__',
            '__str__', '__repr__',
            '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
            '__hash__',
            '__len__', '__getitem__', '__setitem__', '__delitem__',
            '__iter__', '__next__',
            '__call__',
            '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__',
            '__and__', '__or__', '__xor__',
            '__getattr__', '__setattr__', '__delattr__',
            '__enter__', '__exit__',
        ]
    
    def generate_init(self, builder, module, instance_ptr, init_func, args):
        """
        Generate __init__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            init_func: __init__ function pointer
            args: List of initialization arguments
        
        Returns:
            None (modifies instance in place)
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        func = ir.Function(module, func_type, name="call_init")
        
        # Create args array
        args_array = self._create_pointer_array(builder, module, args) if args else ir.Constant(self.void_ptr, None)
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        init_void_ptr = builder.bitcast(init_func, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args) if args else 0)
        
        builder.call(func, [instance_void_ptr, init_void_ptr, num_args, args_array])
    
    def generate_str(self, builder, module, instance_ptr):
        """
        Generate __str__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
        
        Returns:
            String representation
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.char_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="call_str")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        result = builder.call(func, [instance_void_ptr])
        
        return result
    
    def generate_repr(self, builder, module, instance_ptr):
        """
        Generate __repr__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
        
        Returns:
            String representation
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.char_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="call_repr")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        result = builder.call(func, [instance_void_ptr])
        
        return result
    
    def generate_eq(self, builder, module, left_ptr, right_ptr):
        """
        Generate __eq__ method call (equality comparison).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            left_ptr: Left operand
            right_ptr: Right operand
        
        Returns:
            Boolean result (i1)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.int1, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="call_eq")
        
        left_void_ptr = builder.bitcast(left_ptr, self.void_ptr)
        right_void_ptr = builder.bitcast(right_ptr, self.void_ptr)
        
        result = builder.call(func, [left_void_ptr, right_void_ptr])
        
        return result
    
    def generate_hash(self, builder, module, instance_ptr):
        """
        Generate __hash__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
        
        Returns:
            Hash value (i64)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.int64, [self.void_ptr])
        func = ir.Function(module, func_type, name="call_hash")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        result = builder.call(func, [instance_void_ptr])
        
        return result
    
    def generate_len(self, builder, module, instance_ptr):
        """
        Generate __len__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
        
        Returns:
            Length (i64)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.int64, [self.void_ptr])
        func = ir.Function(module, func_type, name="call_len")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        result = builder.call(func, [instance_void_ptr])
        
        return result
    
    def generate_getitem(self, builder, module, instance_ptr, key):
        """
        Generate __getitem__ method call (indexing: obj[key]).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            key: Key/index value
        
        Returns:
            Retrieved value
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="call_getitem")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        key_void_ptr = builder.bitcast(key, self.void_ptr)
        
        result = builder.call(func, [instance_void_ptr, key_void_ptr])
        
        return result
    
    def generate_setitem(self, builder, module, instance_ptr, key, value):
        """
        Generate __setitem__ method call (assignment: obj[key] = value).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            key: Key/index value
            value: Value to set
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="call_setitem")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        key_void_ptr = builder.bitcast(key, self.void_ptr)
        value_void_ptr = builder.bitcast(value, self.void_ptr)
        
        builder.call(func, [instance_void_ptr, key_void_ptr, value_void_ptr])
    
    def generate_iter(self, builder, module, instance_ptr):
        """
        Generate __iter__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
        
        Returns:
            Iterator object
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="call_iter")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        result = builder.call(func, [instance_void_ptr])
        
        return result
    
    def generate_next(self, builder, module, iterator_ptr):
        """
        Generate __next__ method call.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            iterator_ptr: Pointer to Iterator
        
        Returns:
            Next value (raises StopIteration when exhausted)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="call_next")
        
        iterator_void_ptr = builder.bitcast(iterator_ptr, self.void_ptr)
        result = builder.call(func, [iterator_void_ptr])
        
        return result
    
    def generate_call(self, builder, module, instance_ptr, args):
        """
        Generate __call__ method (make object callable).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
            args: List of arguments
        
        Returns:
            Return value
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="call_call")
        
        # Create args array
        args_array = self._create_pointer_array(builder, module, args) if args else ir.Constant(self.void_ptr, None)
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args) if args else 0)
        
        result = builder.call(func, [instance_void_ptr, num_args, args_array])
        
        return result
    
    def generate_binary_op(self, builder, module, op_name, left_ptr, right_ptr):
        """
        Generate binary operator magic method (__add__, __sub__, etc.).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            op_name: Operation name ('__add__', '__sub__', etc.)
            left_ptr: Left operand
            right_ptr: Right operand
        
        Returns:
            Result value
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.char_ptr, self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="call_binary_op")
        
        op_str = self._create_string_literal(builder, module, op_name)
        left_void_ptr = builder.bitcast(left_ptr, self.void_ptr)
        right_void_ptr = builder.bitcast(right_ptr, self.void_ptr)
        
        result = builder.call(func, [op_str, left_void_ptr, right_void_ptr])
        
        return result
    
    def generate_context_manager(self, builder, module, instance_ptr):
        """
        Generate __enter__ and __exit__ for context managers (with statement).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            instance_ptr: Pointer to Instance
        
        Returns:
            Tuple of (enter_result, exit_func)
        """
        # __enter__
        enter_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        enter_func = ir.Function(module, enter_type, name="call_enter")
        
        instance_void_ptr = builder.bitcast(instance_ptr, self.void_ptr)
        enter_result = builder.call(enter_func, [instance_void_ptr])
        
        # __exit__ (will be called later)
        exit_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr, self.void_ptr, self.void_ptr])
        exit_func = ir.Function(module, exit_type, name="call_exit")
        
        return enter_result, exit_func
    
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


def generate_magic_methods_runtime():
    """Generate C runtime code for magic methods."""
    
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

// Helper: Find method by name
void* find_method(Instance* inst, const char* method_name) {
    Class* cls = inst->cls;
    for (int32_t i = 0; i < cls->num_methods; i++) {
        if (strcmp(cls->method_names[i], method_name) == 0) {
            return cls->methods[i];
        }
    }
    return NULL;
}

// __init__ - Constructor
void call_init(void* instance_ptr, void* init_func, int32_t num_args, void* args_array) {
    // Would call __init__ with instance and args
    // For now, no-op
}

// __str__ - String representation
char* call_str(void* instance_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    void* str_method = find_method(inst, "__str__");
    
    if (str_method) {
        // Would call __str__ method
        // For now, return class name
        return inst->cls->name;
    }
    
    return "<object>";
}

// __repr__ - Developer representation
char* call_repr(void* instance_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    void* repr_method = find_method(inst, "__repr__");
    
    if (repr_method) {
        // Would call __repr__ method
        return inst->cls->name;
    }
    
    return "<object>";
}

// __eq__ - Equality
int32_t call_eq(void* left_ptr, void* right_ptr) {
    Instance* left = (Instance*)left_ptr;
    void* eq_method = find_method(left, "__eq__");
    
    if (eq_method) {
        // Would call __eq__ method
        // For now, pointer equality
        return left_ptr == right_ptr;
    }
    
    return left_ptr == right_ptr;
}

// __hash__ - Hash value
int64_t call_hash(void* instance_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    void* hash_method = find_method(inst, "__hash__");
    
    if (hash_method) {
        // Would call __hash__ method
        // For now, pointer hash
        return (int64_t)instance_ptr;
    }
    
    return (int64_t)instance_ptr;
}

// __len__ - Length
int64_t call_len(void* instance_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    void* len_method = find_method(inst, "__len__");
    
    if (len_method) {
        // Would call __len__ method
        return 0;
    }
    
    return 0;
}

// __getitem__ - Indexing
void* call_getitem(void* instance_ptr, void* key) {
    Instance* inst = (Instance*)instance_ptr;
    void* getitem_method = find_method(inst, "__getitem__");
    
    if (getitem_method) {
        // Would call __getitem__ method
        return NULL;
    }
    
    return NULL;
}

// __setitem__ - Assignment
void call_setitem(void* instance_ptr, void* key, void* value) {
    Instance* inst = (Instance*)instance_ptr;
    void* setitem_method = find_method(inst, "__setitem__");
    
    if (setitem_method) {
        // Would call __setitem__ method
    }
}

// __iter__ - Iterator
void* call_iter(void* instance_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    void* iter_method = find_method(inst, "__iter__");
    
    if (iter_method) {
        // Would call __iter__ method
        return instance_ptr;
    }
    
    return NULL;
}

// __next__ - Next value
void* call_next(void* iterator_ptr) {
    Instance* inst = (Instance*)iterator_ptr;
    void* next_method = find_method(inst, "__next__");
    
    if (next_method) {
        // Would call __next__ method
        return NULL;
    }
    
    return NULL;
}

// __call__ - Callable object
void* call_call(void* instance_ptr, int32_t num_args, void* args_array) {
    Instance* inst = (Instance*)instance_ptr;
    void* call_method = find_method(inst, "__call__");
    
    if (call_method) {
        // Would call __call__ method
        return NULL;
    }
    
    return NULL;
}

// Binary operators
void* call_binary_op(char* op_name, void* left_ptr, void* right_ptr) {
    Instance* left = (Instance*)left_ptr;
    void* op_method = find_method(left, op_name);
    
    if (op_method) {
        // Would call operator method
        return NULL;
    }
    
    return NULL;
}

// __enter__ - Context manager entry
void* call_enter(void* instance_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    void* enter_method = find_method(inst, "__enter__");
    
    if (enter_method) {
        // Would call __enter__ method
        return instance_ptr;
    }
    
    return instance_ptr;
}

// __exit__ - Context manager exit
void call_exit(void* instance_ptr, void* exc_type, void* exc_value, void* traceback) {
    Instance* inst = (Instance*)instance_ptr;
    void* exit_method = find_method(inst, "__exit__");
    
    if (exit_method) {
        // Would call __exit__ method
    }
}
"""
    
    # Write to file
    with open('magic_methods_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Magic methods runtime generated: magic_methods_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_magic_methods_runtime()
    
    # Test magic methods
    magic = MagicMethods()
    print(f"✅ MagicMethods initialized")
    print(f"   - Supported magic methods: {len(magic.magic_method_names)}")
    print(f"   - Methods: {', '.join(magic.magic_method_names[:10])}...")
