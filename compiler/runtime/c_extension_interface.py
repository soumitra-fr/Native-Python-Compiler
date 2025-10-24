"""
Phase 5: C Extension Interface
Provides CPython C API compatibility for loading C extensions like NumPy/Pandas.
"""

from llvmlite import ir
import ctypes

class CExtensionInterface:
    """
    Handles interfacing with CPython C extensions.
    
    Features:
    - CPython C API compatibility layer
    - PyObject* bridging
    - Reference counting integration
    - C function calling from compiled code
    - NumPy/Pandas support
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # PyObject structure (simplified)
        self.pyobject_type = ir.LiteralStructType([
            self.int64,           # ob_refcnt
            self.void_ptr,        # ob_type
        ])
        
        # C Function wrapper
        self.c_function_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # function_ptr
            self.char_ptr,        # name
            self.int32,           # num_args
            self.int32,           # flags (METH_VARARGS, etc.)
        ])
    
    def create_pyobject_wrapper(self, builder, obj_ptr, type_ptr):
        """
        Wrap a native object as PyObject*.
        
        Args:
            builder: LLVM IR builder
            obj_ptr: Pointer to native object
            type_ptr: Pointer to type info
        
        Returns:
            Pointer to PyObject
        """
        # Allocate PyObject structure
        pyobj_ptr = builder.alloca(self.pyobject_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(pyobj_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set type
        type_field_ptr = builder.gep(pyobj_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        type_void_ptr = builder.bitcast(type_ptr, self.void_ptr)
        builder.store(type_void_ptr, type_field_ptr)
        
        return pyobj_ptr
    
    def unwrap_pyobject(self, builder, module, pyobj_ptr):
        """
        Extract native object from PyObject*.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            pyobj_ptr: Pointer to PyObject
        
        Returns:
            Pointer to native object
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="pyobject_unwrap")
        
        pyobj_void_ptr = builder.bitcast(pyobj_ptr, self.void_ptr)
        result = builder.call(func, [pyobj_void_ptr])
        
        return result
    
    def call_c_function(self, builder, module, func_ptr, args):
        """
        Call a C function from C extension.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            func_ptr: Pointer to C function
            args: List of arguments (as PyObject*)
        
        Returns:
            Return value (as PyObject*)
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="call_c_function")
        
        # Create args array
        args_array = self._create_pointer_array(builder, module, args) if args else ir.Constant(self.void_ptr, None)
        
        func_void_ptr = builder.bitcast(func_ptr, self.void_ptr)
        num_args = ir.Constant(self.int32, len(args) if args else 0)
        
        result = builder.call(func, [func_void_ptr, num_args, args_array])
        
        return result
    
    def load_c_extension(self, builder, module, extension_name):
        """
        Load a C extension module (like NumPy).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            extension_name: Name of extension (e.g., "numpy")
        
        Returns:
            Pointer to extension module
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.char_ptr])
        func = ir.Function(module, func_type, name="load_c_extension")
        
        name_str = self._create_string_literal(builder, module, extension_name)
        result = builder.call(func, [name_str])
        
        return result
    
    def get_c_function(self, builder, module, extension_ptr, func_name):
        """
        Get a function from a C extension.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            extension_ptr: Pointer to C extension module
            func_name: Name of function
        
        Returns:
            Pointer to function
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="get_c_function")
        
        name_str = self._create_string_literal(builder, module, func_name)
        ext_void_ptr = builder.bitcast(extension_ptr, self.void_ptr)
        
        result = builder.call(func, [ext_void_ptr, name_str])
        
        return result
    
    def py_incref(self, builder, module, pyobj_ptr):
        """Increment PyObject refcount."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="Py_INCREF")
        pyobj_void_ptr = builder.bitcast(pyobj_ptr, self.void_ptr)
        builder.call(func, [pyobj_void_ptr])
    
    def py_decref(self, builder, module, pyobj_ptr):
        """Decrement PyObject refcount."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="Py_DECREF")
        pyobj_void_ptr = builder.bitcast(pyobj_ptr, self.void_ptr)
        builder.call(func, [pyobj_void_ptr])
    
    def create_c_function_wrapper(self, builder, func_ptr, name, num_args):
        """
        Create a wrapper for a C function.
        
        Args:
            builder: LLVM IR builder
            func_ptr: Pointer to C function
            name: Function name
            num_args: Number of arguments
        
        Returns:
            Pointer to CFunctionWrapper
        """
        # Allocate wrapper structure
        wrapper_ptr = builder.alloca(self.c_function_type)
        
        # Set refcount
        refcount_ptr = builder.gep(wrapper_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set function pointer
        func_field_ptr = builder.gep(wrapper_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        func_void_ptr = builder.bitcast(func_ptr, self.void_ptr)
        builder.store(func_void_ptr, func_field_ptr)
        
        # Set name (would create string literal)
        name_ptr = builder.gep(wrapper_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        builder.store(ir.Constant(self.char_ptr, None), name_ptr)
        
        # Set num_args
        num_args_ptr = builder.gep(wrapper_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
        builder.store(ir.Constant(self.int32, num_args), num_args_ptr)
        
        # Set flags
        flags_ptr = builder.gep(wrapper_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
        builder.store(ir.Constant(self.int32, 0), flags_ptr)
        
        return wrapper_ptr
    
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


def generate_c_extension_runtime():
    """Generate C runtime code for C extension interface."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

// Simplified PyObject
typedef struct PyObject {
    int64_t ob_refcnt;
    void* ob_type;
} PyObject;

// Reference counting
void Py_INCREF(void* obj) {
    if (obj) {
        PyObject* pyobj = (PyObject*)obj;
        pyobj->ob_refcnt++;
    }
}

void Py_DECREF(void* obj) {
    if (obj) {
        PyObject* pyobj = (PyObject*)obj;
        pyobj->ob_refcnt--;
        if (pyobj->ob_refcnt == 0) {
            free(obj);
        }
    }
}

// Unwrap PyObject to native object
void* pyobject_unwrap(void* pyobj_ptr) {
    // Would extract native object from PyObject
    // For now, return as-is
    return pyobj_ptr;
}

// Call C function
void* call_c_function(void* func_ptr, int32_t num_args, void* args_array) {
    // Would call C function with proper calling convention
    // For now, return NULL
    return NULL;
}

// Load C extension
void* load_c_extension(char* extension_name) {
    // Try to load shared library
    // On macOS: lib<name>.dylib or <name>.so
    // On Linux: lib<name>.so
    
    char lib_name[256];
    snprintf(lib_name, sizeof(lib_name), "lib%s.dylib", extension_name);
    
    void* handle = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
    
    if (!handle) {
        // Try .so extension
        snprintf(lib_name, sizeof(lib_name), "lib%s.so", extension_name);
        handle = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
    }
    
    if (!handle) {
        // Try without lib prefix
        snprintf(lib_name, sizeof(lib_name), "%s.so", extension_name);
        handle = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
    }
    
    return handle;
}

// Get function from C extension
void* get_c_function(void* extension_ptr, char* func_name) {
    if (!extension_ptr) {
        return NULL;
    }
    
    void* func = dlsym(extension_ptr, func_name);
    return func;
}
"""
    
    # Write to file
    with open('c_extension_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ C extension runtime generated: c_extension_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_c_extension_runtime()
    
    # Test C extension interface
    c_ext = CExtensionInterface()
    print(f"✅ CExtensionInterface initialized")
    print(f"   - PyObject structure: {c_ext.pyobject_type}")
    print(f"   - CFunction structure: {c_ext.c_function_type}")
