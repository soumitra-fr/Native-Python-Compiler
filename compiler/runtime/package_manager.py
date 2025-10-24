"""
Phase 4: Package Manager Implementation
Provides package support with __init__.py, namespaces, and submodules.
"""

from llvmlite import ir
from pathlib import Path
import os

class PackageManager:
    """
    Handles Python packages and namespace management.
    
    Features:
    - __init__.py handling
    - Package namespaces
    - Submodule imports
    - __all__ attribute support
    - __path__ attribute
    """
    
    def __init__(self, module_loader):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        self.module_loader = module_loader
        
        # Package structure
        self.package_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.char_ptr,        # name
            self.char_ptr,        # path (__path__)
            self.void_ptr,        # dict (namespace)
            self.void_ptr,        # submodules (dict of submodules)
            self.void_ptr,        # __all__ (list of exported names)
        ])
    
    def is_package(self, path):
        """
        Check if a directory is a package (contains __init__.py).
        
        Args:
            path: Directory path
        
        Returns:
            True if package, False otherwise
        """
        init_file = Path(path) / '__init__.py'
        return init_file.exists()
    
    def get_package_path(self, package_name, search_paths=None):
        """
        Get the path to a package directory.
        
        Args:
            package_name: Name of package
            search_paths: Optional search paths
        
        Returns:
            Package directory path or None
        """
        if search_paths is None:
            import sys
            search_paths = sys.path
        
        parts = package_name.split('.')
        
        for base_path in search_paths:
            pkg_path = Path(base_path) / Path(*parts)
            if pkg_path.exists() and self.is_package(pkg_path):
                return str(pkg_path)
        
        return None
    
    def create_package(self, builder, module, package_name, package_path):
        """
        Create a Package structure in LLVM IR.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            package_name: Name of package
            package_path: Path to package directory
        
        Returns:
            Pointer to Package structure
        """
        # Allocate package structure
        pkg_ptr = builder.alloca(self.package_type)
        
        # Set refcount to 1
        refcount_ptr = builder.gep(pkg_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 0)])
        builder.store(ir.Constant(self.int64, 1), refcount_ptr)
        
        # Set name
        name_str = self._create_string_literal(builder, module, package_name)
        name_ptr = builder.gep(pkg_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 1)])
        builder.store(name_str, name_ptr)
        
        # Set path (__path__)
        path_str = self._create_string_literal(builder, module, package_path)
        path_ptr = builder.gep(pkg_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 2)])
        builder.store(path_str, path_ptr)
        
        # Initialize empty dict (namespace)
        dict_ptr = builder.gep(pkg_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 3)])
        builder.store(ir.Constant(self.void_ptr, None), dict_ptr)
        
        # Initialize empty submodules dict
        submodules_ptr = builder.gep(pkg_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 4)])
        builder.store(ir.Constant(self.void_ptr, None), submodules_ptr)
        
        # Initialize __all__ to None (will be set if defined in __init__.py)
        all_ptr = builder.gep(pkg_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, 5)])
        builder.store(ir.Constant(self.void_ptr, None), all_ptr)
        
        return pkg_ptr
    
    def load_submodule(self, builder, module, package_ptr, submodule_name):
        """
        Load a submodule of a package.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            package_ptr: Pointer to Package
            submodule_name: Name of submodule (e.g., "utils" for "package.utils")
        
        Returns:
            Pointer to submodule
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="package_load_submodule")
        
        name_str = self._create_string_literal(builder, module, submodule_name)
        package_void_ptr = builder.bitcast(package_ptr, self.void_ptr)
        
        result = builder.call(func, [package_void_ptr, name_str])
        
        return result
    
    def get_all_exports(self, builder, module, package_ptr):
        """
        Get __all__ list from package (names to export with import *).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            package_ptr: Pointer to Package
        
        Returns:
            Pointer to __all__ list (or None if not defined)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="package_get_all")
        
        package_void_ptr = builder.bitcast(package_ptr, self.void_ptr)
        result = builder.call(func, [package_void_ptr])
        
        return result
    
    def set_all_exports(self, builder, module, package_ptr, names):
        """
        Set __all__ list for package.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            package_ptr: Pointer to Package
            names: List of names to export
        """
        # Declare runtime function
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="package_set_all")
        
        # Create list of names (would use list type from Phase 1)
        names_ptr = ir.Constant(self.void_ptr, None)  # Placeholder
        
        package_void_ptr = builder.bitcast(package_ptr, self.void_ptr)
        builder.call(func, [package_void_ptr, names_ptr])
    
    def list_submodules(self, package_path):
        """
        List all submodules in a package directory.
        
        Args:
            package_path: Path to package directory
        
        Returns:
            List of submodule names
        """
        if not os.path.exists(package_path):
            return []
        
        submodules = []
        
        for item in os.listdir(package_path):
            item_path = os.path.join(package_path, item)
            
            # Check if it's a submodule (.py file)
            if item.endswith('.py') and item != '__init__.py':
                submodules.append(item[:-3])  # Remove .py extension
            
            # Check if it's a subpackage (directory with __init__.py)
            elif os.path.isdir(item_path) and self.is_package(item_path):
                submodules.append(item)
        
        return sorted(submodules)
    
    def get_init_file(self, package_path):
        """
        Get the __init__.py file path for a package.
        
        Args:
            package_path: Path to package directory
        
        Returns:
            Path to __init__.py or None
        """
        init_file = Path(package_path) / '__init__.py'
        return str(init_file) if init_file.exists() else None
    
    def incref(self, builder, module, package_ptr):
        """Increment reference count."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="package_incref")
        package_void_ptr = builder.bitcast(package_ptr, self.void_ptr)
        builder.call(func, [package_void_ptr])
    
    def decref(self, builder, module, package_ptr):
        """Decrement reference count and free if zero."""
        func_type = ir.FunctionType(ir.VoidType(), [self.void_ptr])
        func = ir.Function(module, func_type, name="package_decref")
        package_void_ptr = builder.bitcast(package_ptr, self.void_ptr)
        builder.call(func, [package_void_ptr])
    
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


def generate_package_manager_runtime():
    """Generate C runtime code for package management."""
    
    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Package structure
typedef struct Package {
    int64_t refcount;
    char* name;
    char* path;           // __path__
    void* dict;           // Package namespace
    void* submodules;     // Dict of submodules
    void* all_list;       // __all__ list
} Package;

// Reference counting
void package_incref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)++;
    }
}

void package_decref(void* obj) {
    if (obj) {
        int64_t* refcount = (int64_t*)obj;
        (*refcount)--;
        if (*refcount == 0) {
            Package* pkg = (Package*)obj;
            if (pkg->name) free(pkg->name);
            if (pkg->path) free(pkg->path);
            free(obj);
        }
    }
}

// Load submodule
void* package_load_submodule(void* package_ptr, char* submodule_name) {
    Package* pkg = (Package*)package_ptr;
    
    // Would load submodule from package path
    // For now, return NULL
    return NULL;
}

// Get __all__ list
void* package_get_all(void* package_ptr) {
    Package* pkg = (Package*)package_ptr;
    return pkg->all_list;
}

// Set __all__ list
void package_set_all(void* package_ptr, void* names_list) {
    Package* pkg = (Package*)package_ptr;
    pkg->all_list = names_list;
}
"""
    
    # Write to file
    with open('package_manager_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ Package manager runtime generated: package_manager_runtime.c")


if __name__ == "__main__":
    from module_loader import ModuleLoader
    
    # Generate runtime C code
    generate_package_manager_runtime()
    
    # Test package manager
    loader = ModuleLoader()
    pkg_mgr = PackageManager(loader)
    
    print(f"✅ PackageManager initialized")
    print(f"   - Package structure: {pkg_mgr.package_type}")
    
    # Test package detection
    import sys
    test_packages = ['os', 'json', 'email']
    for pkg_name in test_packages:
        path = pkg_mgr.get_package_path(pkg_name)
        if path:
            print(f"   - Found package '{pkg_name}': {path}")
            submodules = pkg_mgr.list_submodules(path)
            print(f"     Submodules: {len(submodules)} found")
