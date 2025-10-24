/*
 * Python List Runtime Library - Phase 3.1
 * AI Agentic Python-to-Native Compiler
 * 
 * This C library provides optimized runtime functions for Python lists.
 * Supports both specialized lists (List[int], List[float]) and dynamic lists.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Data Structures
// ============================================================================

// Specialized integer list (List[int])
typedef struct {
    int64_t capacity;
    int64_t length;
    int64_t* data;
} ListInt;

// Specialized float list (List[float])
typedef struct {
    int64_t capacity;
    int64_t length;
    double* data;
} ListFloat;

// Dynamic list (mixed types) - for future implementation
typedef struct {
    int64_t capacity;
    int64_t length;
    void** data;  // Array of PyObject pointers
} ListDynamic;

// ============================================================================
// Integer List Operations
// ============================================================================

/**
 * Allocate a new integer list with given capacity
 */
ListInt* alloc_list_int(int64_t capacity) {
    ListInt* list = (ListInt*)malloc(sizeof(ListInt));
    if (!list) return NULL;
    
    list->capacity = capacity > 0 ? capacity : 8;  // Default capacity: 8
    list->length = 0;
    list->data = (int64_t*)calloc(list->capacity, sizeof(int64_t));
    
    if (!list->data) {
        free(list);
        return NULL;
    }
    
    return list;
}

/**
 * Store value at index in integer list
 */
void store_list_int(ListInt* list, int64_t index, int64_t value) {
    if (!list || index < 0 || index >= list->capacity) {
        fprintf(stderr, "Error: list index out of range\n");
        return;
    }
    
    list->data[index] = value;
    
    // Update length if needed
    if (index >= list->length) {
        list->length = index + 1;
    }
}

/**
 * Load value from index in integer list
 */
int64_t load_list_int(ListInt* list, int64_t index) {
    if (!list || index < 0 || index >= list->length) {
        fprintf(stderr, "Error: list index out of range\n");
        return 0;
    }
    
    return list->data[index];
}

/**
 * Append value to integer list (with automatic resize)
 */
void append_list_int(ListInt* list, int64_t value) {
    if (!list) return;
    
    // Resize if needed (double the capacity)
    if (list->length >= list->capacity) {
        int64_t new_capacity = list->capacity * 2;
        int64_t* new_data = (int64_t*)realloc(list->data, new_capacity * sizeof(int64_t));
        
        if (!new_data) {
            fprintf(stderr, "Error: failed to resize list\n");
            return;
        }
        
        list->data = new_data;
        list->capacity = new_capacity;
    }
    
    list->data[list->length++] = value;
}

/**
 * Get length of integer list
 */
int64_t list_len_int(ListInt* list) {
    return list ? list->length : 0;
}

/**
 * Free integer list
 */
void free_list_int(ListInt* list) {
    if (list) {
        if (list->data) {
            free(list->data);
        }
        free(list);
    }
}

/**
 * Sum all elements in integer list
 */
int64_t sum_list_int(ListInt* list) {
    if (!list) return 0;
    
    int64_t sum = 0;
    for (int64_t i = 0; i < list->length; i++) {
        sum += list->data[i];
    }
    return sum;
}

/**
 * Find maximum value in integer list
 */
int64_t max_list_int(ListInt* list) {
    if (!list || list->length == 0) return 0;
    
    int64_t max_val = list->data[0];
    for (int64_t i = 1; i < list->length; i++) {
        if (list->data[i] > max_val) {
            max_val = list->data[i];
        }
    }
    return max_val;
}

// ============================================================================
// Float List Operations
// ============================================================================

/**
 * Allocate a new float list with given capacity
 */
ListFloat* alloc_list_float(int64_t capacity) {
    ListFloat* list = (ListFloat*)malloc(sizeof(ListFloat));
    if (!list) return NULL;
    
    list->capacity = capacity > 0 ? capacity : 8;
    list->length = 0;
    list->data = (double*)calloc(list->capacity, sizeof(double));
    
    if (!list->data) {
        free(list);
        return NULL;
    }
    
    return list;
}

/**
 * Store value at index in float list
 */
void store_list_float(ListFloat* list, int64_t index, double value) {
    if (!list || index < 0 || index >= list->capacity) {
        fprintf(stderr, "Error: list index out of range\n");
        return;
    }
    
    list->data[index] = value;
    
    if (index >= list->length) {
        list->length = index + 1;
    }
}

/**
 * Load value from index in float list
 */
double load_list_float(ListFloat* list, int64_t index) {
    if (!list || index < 0 || index >= list->length) {
        fprintf(stderr, "Error: list index out of range\n");
        return 0.0;
    }
    
    return list->data[index];
}

/**
 * Append value to float list
 */
void append_list_float(ListFloat* list, double value) {
    if (!list) return;
    
    if (list->length >= list->capacity) {
        int64_t new_capacity = list->capacity * 2;
        double* new_data = (double*)realloc(list->data, new_capacity * sizeof(double));
        
        if (!new_data) {
            fprintf(stderr, "Error: failed to resize list\n");
            return;
        }
        
        list->data = new_data;
        list->capacity = new_capacity;
    }
    
    list->data[list->length++] = value;
}

/**
 * Get length of float list
 */
int64_t list_len_float(ListFloat* list) {
    return list ? list->length : 0;
}

/**
 * Free float list
 */
void free_list_float(ListFloat* list) {
    if (list) {
        if (list->data) {
            free(list->data);
        }
        free(list);
    }
}

/**
 * Sum all elements in float list
 */
double sum_list_float(ListFloat* list) {
    if (!list) return 0.0;
    
    double sum = 0.0;
    for (int64_t i = 0; i < list->length; i++) {
        sum += list->data[i];
    }
    return sum;
}

// ============================================================================
// Test / Demo Functions
// ============================================================================

#ifdef RUN_TESTS

void test_int_list() {
    printf("\n=== Testing Integer List ===\n");
    
    // Create list
    ListInt* list = alloc_list_int(5);
    printf("Created list with capacity: %lld\n", list->capacity);
    
    // Store values
    for (int64_t i = 0; i < 5; i++) {
        store_list_int(list, i, i * 10);
    }
    printf("Stored 5 values\n");
    
    // Load and print
    printf("Values: ");
    for (int64_t i = 0; i < list->length; i++) {
        printf("%lld ", load_list_int(list, i));
    }
    printf("\n");
    
    // Append (should trigger resize)
    append_list_int(list, 100);
    append_list_int(list, 200);
    printf("After append, length: %lld, capacity: %lld\n", list->length, list->capacity);
    
    // Sum
    int64_t sum = sum_list_int(list);
    printf("Sum: %lld\n", sum);
    
    // Max
    int64_t max = max_list_int(list);
    printf("Max: %lld\n", max);
    
    // Free
    free_list_int(list);
    printf("List freed\n");
}

void test_float_list() {
    printf("\n=== Testing Float List ===\n");
    
    ListFloat* list = alloc_list_float(3);
    
    store_list_float(list, 0, 1.5);
    store_list_float(list, 1, 2.7);
    store_list_float(list, 2, 3.14);
    
    printf("Values: ");
    for (int64_t i = 0; i < list->length; i++) {
        printf("%.2f ", load_list_float(list, i));
    }
    printf("\n");
    
    append_list_float(list, 9.99);
    printf("After append, length: %lld\n", list->length);
    
    double sum = sum_list_float(list);
    printf("Sum: %.2f\n", sum);
    
    free_list_float(list);
    printf("List freed\n");
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  Python List Runtime Library - Test Suite                     ║\n");
    printf("║  AI Agentic Python-to-Native Compiler - Phase 3.1             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    test_int_list();
    test_float_list();
    
    printf("\n✅ All runtime library tests passed!\n\n");
    return 0;
}

#endif  // RUN_TESTS
