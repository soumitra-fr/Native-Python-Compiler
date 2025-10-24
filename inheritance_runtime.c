
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

// MRO-based method lookup
void* mro_lookup_method(void* class_ptr, char* method_name) {
    Class* cls = (Class*)class_ptr;
    
    // Search in current class
    for (int32_t i = 0; i < cls->num_methods; i++) {
        if (strcmp(cls->method_names[i], method_name) == 0) {
            return cls->methods[i];
        }
    }
    
    // Search in base classes (left-to-right, depth-first)
    for (int32_t i = 0; i < cls->num_bases; i++) {
        void* method = mro_lookup_method(cls->bases[i], method_name);
        if (method) {
            return method;
        }
    }
    
    return NULL;
}

// Super call - find method in parent class
void* super_call_method(void* instance_ptr, void* current_class_ptr, 
                       char* method_name, int32_t num_args, void* args_array) {
    Instance* inst = (Instance*)instance_ptr;
    Class* current_cls = (Class*)current_class_ptr;
    
    // Find current class in instance's class hierarchy
    // Then look in next class in MRO
    
    // Simplified: just look in first base
    if (current_cls->num_bases > 0) {
        return mro_lookup_method(current_cls->bases[0], method_name);
    }
    
    return NULL;
}

// isinstance check
int32_t isinstance_check(void* instance_ptr, void* class_ptr) {
    Instance* inst = (Instance*)instance_ptr;
    Class* target_cls = (Class*)class_ptr;
    Class* inst_cls = inst->cls;
    
    // Check if instance's class is the target class
    if (inst_cls == target_cls) {
        return 1;
    }
    
    // Check if target class is in instance's class MRO
    for (int32_t i = 0; i < inst_cls->num_bases; i++) {
        if (isinstance_check(instance_ptr, inst_cls->bases[i])) {
            return 1;
        }
    }
    
    return 0;
}

// issubclass check
int32_t issubclass_check(void* subclass_ptr, void* superclass_ptr) {
    Class* sub = (Class*)subclass_ptr;
    Class* super = (Class*)superclass_ptr;
    
    // Check if subclass is superclass
    if (sub == super) {
        return 1;
    }
    
    // Check if superclass is in subclass's bases
    for (int32_t i = 0; i < sub->num_bases; i++) {
        if (issubclass_check(sub->bases[i], superclass_ptr)) {
            return 1;
        }
    }
    
    return 0;
}

// Multiple inheritance attribute lookup
void* multiple_inheritance_lookup(void* class_ptr, char* attr_name) {
    Class* cls = (Class*)class_ptr;
    
    // Search in current class
    for (int32_t i = 0; i < cls->num_attrs; i++) {
        if (strcmp(cls->attr_names[i], attr_name) == 0) {
            return cls->attributes[i];
        }
    }
    
    // Search in base classes (C3 linearization order)
    // Simplified: left-to-right
    for (int32_t i = 0; i < cls->num_bases; i++) {
        void* attr = multiple_inheritance_lookup(cls->bases[i], attr_name);
        if (attr) {
            return attr;
        }
    }
    
    return NULL;
}
