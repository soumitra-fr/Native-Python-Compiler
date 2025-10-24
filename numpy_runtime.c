
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// Simplified NumPy array structure
typedef struct NDArray {
    int64_t refcount;
    void* data;
    int32_t ndim;
    int64_t* shape;
    int64_t* strides;
    int32_t dtype;
    int32_t itemsize;
    int64_t size;
} NDArray;

// Create NumPy array
void* numpy_create_array(int32_t ndim, void* shape_ptr, char* dtype) {
    NDArray* arr = (NDArray*)malloc(sizeof(NDArray));
    arr->refcount = 1;
    arr->ndim = ndim;
    
    // Copy shape
    arr->shape = (int64_t*)malloc(ndim * sizeof(int64_t));
    memcpy(arr->shape, shape_ptr, ndim * sizeof(int64_t));
    
    // Calculate total size
    arr->size = 1;
    for (int32_t i = 0; i < ndim; i++) {
        arr->size *= arr->shape[i];
    }
    
    // Set itemsize based on dtype
    arr->itemsize = 8;  // float64 by default
    
    // Allocate data
    arr->data = calloc(arr->size, arr->itemsize);
    
    // Calculate strides (C-contiguous)
    arr->strides = (int64_t*)malloc(ndim * sizeof(int64_t));
    int64_t stride = arr->itemsize;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = stride;
        stride *= arr->shape[i];
    }
    
    return arr;
}

// Get data pointer
void* numpy_get_data(void* array_ptr) {
    NDArray* arr = (NDArray*)array_ptr;
    return arr->data;
}

// Get shape
void* numpy_get_shape(void* array_ptr) {
    NDArray* arr = (NDArray*)array_ptr;
    return arr->shape;
}

// Get item
double numpy_getitem(void* array_ptr, int32_t num_indices, void* indices_ptr) {
    NDArray* arr = (NDArray*)array_ptr;
    int64_t* indices = (int64_t*)indices_ptr;
    
    // Calculate flat index
    int64_t flat_index = 0;
    int64_t stride = 1;
    for (int32_t i = arr->ndim - 1; i >= 0; i--) {
        flat_index += indices[i] * stride;
        stride *= arr->shape[i];
    }
    
    double* data = (double*)arr->data;
    return data[flat_index];
}

// Set item
void numpy_setitem(void* array_ptr, int32_t num_indices, void* indices_ptr, double value) {
    NDArray* arr = (NDArray*)array_ptr;
    int64_t* indices = (int64_t*)indices_ptr;
    
    // Calculate flat index
    int64_t flat_index = 0;
    int64_t stride = 1;
    for (int32_t i = arr->ndim - 1; i >= 0; i--) {
        flat_index += indices[i] * stride;
        stride *= arr->shape[i];
    }
    
    double* data = (double*)arr->data;
    data[flat_index] = value;
}

// Call ufunc
void* numpy_call_ufunc(char* ufunc_name, int32_t num_arrays, void* arrays_ptr) {
    // Would call actual NumPy ufunc
    // For now, return NULL
    return NULL;
}

// Dot product
void* numpy_dot(void* array1_ptr, void* array2_ptr) {
    // Would perform matrix multiplication
    // For now, return NULL
    return NULL;
}

// Sum
double numpy_sum(void* array_ptr, int32_t axis) {
    NDArray* arr = (NDArray*)array_ptr;
    double* data = (double*)arr->data;
    
    double sum = 0.0;
    for (int64_t i = 0; i < arr->size; i++) {
        sum += data[i];
    }
    
    return sum;
}

// Reshape
void* numpy_reshape(void* array_ptr, int32_t new_ndim, void* new_shape_ptr) {
    // Would reshape array
    // For now, return original array
    return array_ptr;
}
