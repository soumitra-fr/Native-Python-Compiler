
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
