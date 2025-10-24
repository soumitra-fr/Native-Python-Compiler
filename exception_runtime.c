
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>

// Exception structure
typedef struct {
    int64_t refcount;
    int32_t type_id;
    char* message;
} Exception;

// Thread-local exception state
static Exception* current_exception = NULL;
static jmp_buf exception_jump_buffer;
static int exception_jump_set = 0;

// Create new exception
Exception* exception_new(int32_t type_id, const char* message) {
    Exception* exc = (Exception*)malloc(sizeof(Exception));
    exc->refcount = 1;
    exc->type_id = type_id;
    exc->message = strdup(message);
    return exc;
}

// Increment reference count
void exception_incref(Exception* exc) {
    if (exc) exc->refcount++;
}

// Decrement reference count
void exception_decref(Exception* exc) {
    if (exc && --exc->refcount == 0) {
        free(exc->message);
        free(exc);
    }
}

// Raise exception
void exception_raise(Exception* exc) {
    if (current_exception) {
        exception_decref(current_exception);
    }
    current_exception = exc;
    exception_incref(exc);
    
    // If we have a jump buffer set, jump to exception handler
    if (exception_jump_set) {
        longjmp(exception_jump_buffer, 1);
    }
    // Otherwise, this is an unhandled exception (will crash)
}

// Get current exception
Exception* exception_get_current() {
    return current_exception;
}

// Clear current exception
void exception_clear() {
    if (current_exception) {
        exception_decref(current_exception);
        current_exception = NULL;
    }
}

// Check if exception matches type
int32_t exception_matches(Exception* exc, int32_t type_id) {
    if (!exc) return 0;
    // In full implementation, would check inheritance hierarchy
    return exc->type_id == type_id || type_id == 0;  // 0 = Exception (catches all)
}

// Set exception handler jump point
int exception_set_handler() {
    exception_jump_set = 1;
    return setjmp(exception_jump_buffer);
}

// Clear exception handler
void exception_clear_handler() {
    exception_jump_set = 0;
}
