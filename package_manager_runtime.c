
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
