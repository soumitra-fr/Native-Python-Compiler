
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Closure structure
typedef struct {
    int64_t refcount;
    int64_t num_vars;
    void** vars;
} Closure;

// Create new closure
Closure* closure_new(int64_t num_vars) {
    Closure* closure = (Closure*)malloc(sizeof(Closure));
    closure->refcount = 1;
    closure->num_vars = num_vars;
    closure->vars = (void**)malloc(sizeof(void*) * num_vars);
    memset(closure->vars, 0, sizeof(void*) * num_vars);
    return closure;
}

// Set captured variable
void closure_set_var(Closure* closure, int64_t index, void* value) {
    if (index >= 0 && index < closure->num_vars) {
        closure->vars[index] = value;
    }
}

// Get captured variable
void* closure_get_var(Closure* closure, int64_t index) {
    if (index >= 0 && index < closure->num_vars) {
        return closure->vars[index];
    }
    return NULL;
}

// Increment reference count
void closure_incref(Closure* closure) {
    if (closure) closure->refcount++;
}

// Decrement reference count
void closure_decref(Closure* closure) {
    if (closure && --closure->refcount == 0) {
        free(closure->vars);
        free(closure);
    }
}
