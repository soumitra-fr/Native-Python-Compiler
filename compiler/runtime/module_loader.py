"""
Phase 4: Module Loader Implementation
Provides module loading, caching, and search path management.
"""

from llvmlite import ir
import sys
import os
import importlib.util
from pathlib import Path

class ModuleLoader:
    """
    Handles Python module loading for compiled code.
    
    Features:
    - Module search paths (sys.path)
    - Module caching (sys.modules)
    - Circular import detection
    - Module initialization
    - Namespace management
    """
    
    def __init__(self):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        # Module structure
        self.module_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.char_ptr,        # name
            self.char_ptr,        # filename
            self.void_ptr,        # dict (module namespace)
            self.void_ptr,        # parent (parent module)
            self.int32,           # is_package
            self.int32,           # is_loaded
        ])
        
        # Module cache (tracks loaded modules)
        self.loaded_modules = {}
        self.loading_modules = set()  # For circular import detection
    
    def find_module(self, module_name, search_paths=None):
        """
        Find a module by name in the search paths.
        
        Args:
            module_name: Name of module (e.g., "numpy", "mypackage.submodule")
            search_paths: List of paths to search (defaults to sys.path)
        
        Returns:
            Module file path or None if not found
        """
        if search_paths is None:
            search_paths = sys.path
        
        # Replace dots with path separators for nested modules
        parts = module_name.split('.')
        
        for base_path in search_paths:
            # Try as package (__init__.py)
            package_path = Path(base_path) / Path(*parts) / '__init__.py'
            if package_path.exists():
                return str(package_path), True  # is_package=True
            
            # Try as module (.py file)
            module_path = Path(base_path) / Path(*parts[:-1]) / (parts[-1] + '.py')
            if module_path.exists():
                return str(module_path), False  # is_package=False
        
        return None, False
    
    def load_module(self, builder, module, module_name, search_paths=None):
        """
        Load a module by name.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_name: Name of module to load
            search_paths: Optional custom search paths
        
        Returns:
            Pointer to Module structure
        """
        # Check if already loaded
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        # Check for circular imports
        if module_name in self.loading_modules:
            raise ImportError(f"Circular import detected: {module_name}")
        
        # Mark as loading
        self.loading_modules.add(module_name)
        
        try:
            # Find module
            module_path, is_package = self.find_module(module_name, search_paths)
            
            if module_path is None:
                raise ImportError(f"No module named '{module_name}'")
            
            # Create module structure
            module_ptr = self.create_module_object(builder, module, module_name, module_path, is_package)
            
            # Cache the module
            self.loaded_modules[module_name] = module_ptr
            
            # Mark as loaded
            return module_ptr
        
        finally:
            # Remove from loading set
            self.loading_modules.discard(module_name)
    
    def create_module_object(self, builder, module, module_name, module_path, is_package):
        """
        Create a Module structure in LLVM IR.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_name: Name of the module
            module_path: Path to module file
            is_package: Whether this is a package (has __init__.py)
        
        Returns:
            Pointer to Module structure
        """
        # Allocate module structure
        module_ptr = builder.alloca(self.module_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set name
        name_str = self._create_string_literal(builder, module, module_name)
        name_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        builder.store(name_str, name_ptr)
        
        # Set filename
        filename_str = self._create_string_literal(builder, module, module_path)
        filename_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        builder.store(filename_str, filename_ptr)
        
        # Set dict (namespace) - initialized by runtime
        dict_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
        builder.store(ir.Constant(self.void_ptr, None), dict_ptr)
        
        # Set parent (for nested modules)
        parent_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
        builder.store(ir.Constant(self.void_ptr, None), parent_ptr)
        
        # Set is_package
        is_package_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 5)])
        builder.store(ir.Constant(self.int32, 1 if is_package else 0), is_package_ptr)
        
        # Set is_loaded to 0 (will be set to 1 after initialization)
        is_loaded_ptr = builder.gep(module_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 6)])
        builder.store(ir.Constant(self.int32, 0), is_loaded_ptr)
        
        return module_ptr
    
    def get_module_attribute(self, builder, module, module_ptr, attr_name):
        """
        Get an attribute from a module's namespace.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_ptr: Pointer to Module structure
            attr_name: Name of attribute
        
        Returns:
            Attribute value (void*)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="module_get_attr")
        
        attr_str = self._create_string_literal(builder, module, attr_name)
        module_void_ptr = builder.bitcast(module_ptr, self.void_ptr)
        
        result = builder.call(func, [module_void_ptr, attr_str])
        
        return result
    
    def set_module_attribute(self, builder, module, module_ptr, attr_name, value):
        """
        Set an attribute in a module's namespace.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_ptr: Pointer to Module structure
            attr_name: Name of attribute
            value: Value to set
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.char_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="module_set_attr")
        
        attr_str = self._create_string_literal(builder, module, attr_name)
        module_void_ptr = builder.bitcast(module_ptr, self.void_ptr)
        value_void_ptr = builder.bitcast(value, self.void_ptr)
        
        builder.call(func, [module_void_ptr, attr_str, value_void_ptr])
    
    def reload_module(self, builder, module, module_ptr):
        """
        Reload a module (re-execute module code).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            module_ptr: Pointer to Module structure
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="module_reload")
        
        module_void_ptr = builder.bitcast(module_ptr, self.void_ptr)
        builder.call(func, [module_void_ptr])
    
    def get_search_paths(self):
        """Get current module search paths (sys.path)."""
        return sys.path.copy()
    
    def add_search_path(self, path):
        """Add a path to module search paths."""
        if path not in sys.path:
            sys.path.insert(0, path)
    
    def clear_cache(self):
        """Clear the module cache."""
        self.loaded_modules.clear()
        self.loading_modules.clear()
    
    def incref(self, builder, module, module_ptr):
        """Increment reference count."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="module_incref")
        module_void_ptr = builder.bitcast(module_ptr, self.void_ptr)
        builder.call(func, [module_void_ptr])
    
    def decref(self, builder, module, module_ptr):
        """Decrement reference count and free if zero."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="module_decref")
        module_void_ptr = builder.bitcast(module_ptr, self.void_ptr)
        builder.call(func, [module_void_ptr])
    
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


def generate_module_loader_runtime():
    """Generate C runtime code for module loading."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Module structure
typedef struct Module {
    int64_t refcount;
    char* name;
    char* filename;
    void* dict;           // Module namespace (dict)
    void* parent;         // Parent module
    int32_t is_package;
    int32_t is_loaded;
} Module;

// Reference counting
void module_incref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)++;
    }
}

void module_decref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)--;
        if (*refcount == 0) {
            Module* mod = (Module*)obj;
            if (mod->name) free(mod->name);
            if (mod->filename) free(mod->filename);
            free(obj);
        }
    }
}

// Get attribute from module namespace
void* module_get_attr(void* module_ptr, char* attr_name) {
    Module* mod = (Module*)module_ptr;
    
    if (!mod->dict) {
        return NULL;  // AttributeError
    }
    
    // Would look up in module dict
    // For now, return NULL
    return NULL;
}

// Set attribute in module namespace
void module_set_attr(void* module_ptr, char* attr_name, void* value) {
    Module* mod = (Module*)module_ptr;
    
    if (!mod->dict) {
        // Create dict if not exists
        // For now, no-op
        return;
    }
    
    // Would set in module dict
}

// Reload module
void module_reload(void* module_ptr) {
    Module* mod = (Module*)module_ptr;
    
    // Would re-execute module code
    // For now, no-op
}
"""
    
    # Write to file
    with open('module_loader_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Module loader runtime generated: module_loader_runtime.c")


if __name__ == "__main__":
    # Generate runtime C code
    generate_module_loader_runtime()
    
    # Test module loader
    loader = ModuleLoader()
    print(f"✅ ModuleLoader initialized")
    print(f"   - Module structure: {loader.module_type}")
    print(f"   - Search paths: {len(loader.get_search_paths())} paths")
    
    # Test finding a real module
    try:
        path, is_pkg = loader.find_module("os")
        if path:
            print(f"   - Found 'os' module: {path} (package: {is_pkg})")
    except:
        print(f"   - Could not find 'os' module")
