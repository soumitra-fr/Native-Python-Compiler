
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
