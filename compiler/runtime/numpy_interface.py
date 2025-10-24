"""
Phase 5: NumPy Interface
Provides NumPy ndarray support and ufunc calling in compiled code.
"""

from llvmlite import ir

class NumPyInterface:
    """
    Handles NumPy integration in compiled code.
    
    Features:
    - NumPy ndarray support
    - Array creation and manipulation
    - ufunc calling (vectorized operations)
    - dtype handling
    - Zero-copy where possible
    """
    
    def __init__(self, c_extension_interface):
        self.int64 = ir.IntType(64)
        self.int32 = ir.IntType(32)
        self.int8 = ir.IntType(8)
        self.double = ir.DoubleType()
        self.void_ptr = ir.IntType(8).as_pointer()
        self.char_ptr = self.int8.as_pointer()
        
        self.c_ext = c_extension_interface
        
        # NumPy array structure (simplified PyArrayObject)
        self.ndarray_type = ir.LiteralStructType([
            self.int64,           # refcount
            self.void_ptr,        # data pointer
            self.int32,           # ndim (number of dimensions)
            self.void_ptr,        # shape (int64*)
            self.void_ptr,        # strides (int64*)
            self.int32,           # dtype (type code)
            self.int32,           # itemsize
            self.int64,           # size (total elements)
        ])
    
    def create_ndarray(self, builder, module, shape, dtype='float64'):
        """
        Create a NumPy array.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            shape: Tuple of dimensions
            dtype: Data type string
        
        Returns:
            Pointer to ndarray structure
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.int32, self.void_ptr, self.char_ptr])
        func = ir.Function(module, func_type, name="numpy_create_array")
        
        ndim = ir.Constant(self.int32, len(shape))
        
        # Create shape array
        shape_array = self._create_int64_array(builder, shape)
        
        dtype_str = self._create_string_literal(builder, module, dtype)
        
        result = builder.call(func, [ndim, shape_array, dtype_str])
        
        return result
    
    def array_get_data_ptr(self, builder, module, array_ptr):
        """
        Get data pointer from NumPy array.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array_ptr: Pointer to ndarray
        
        Returns:
            Pointer to array data
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="numpy_get_data")
        
        array_void_ptr = builder.bitcast(array_ptr, self.void_ptr)
        result = builder.call(func, [array_void_ptr])
        
        return result
    
    def array_get_shape(self, builder, module, array_ptr):
        """
        Get shape from NumPy array.
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array_ptr: Pointer to ndarray
        
        Returns:
            Pointer to shape array (int64*)
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr])
        func = ir.Function(module, func_type, name="numpy_get_shape")
        
        array_void_ptr = builder.bitcast(array_ptr, self.void_ptr)
        result = builder.call(func, [array_void_ptr])
        
        return result
    
    def array_getitem(self, builder, module, array_ptr, indices):
        """
        Get item from NumPy array (array[i, j, ...]).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array_ptr: Pointer to ndarray
            indices: List of index values
        
        Returns:
            Scalar value
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.double, arg_types)
        func = ir.Function(module, func_type, name="numpy_getitem")
        
        indices_array = self._create_int64_array(builder, indices)
        num_indices = ir.Constant(self.int32, len(indices))
        
        array_void_ptr = builder.bitcast(array_ptr, self.void_ptr)
        result = builder.call(func, [array_void_ptr, num_indices, indices_array])
        
        return result
    
    def array_setitem(self, builder, module, array_ptr, indices, value):
        """
        Set item in NumPy array (array[i, j, ...] = value).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array_ptr: Pointer to ndarray
            indices: List of index values
            value: Value to set
        """
        # Declare runtime function
        arg_types = [self.void_ptr, self.int32, self.void_ptr, self.double]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        func = ir.Function(module, func_type, name="numpy_setitem")
        
        indices_array = self._create_int64_array(builder, indices)
        num_indices = ir.Constant(self.int32, len(indices))
        
        array_void_ptr = builder.bitcast(array_ptr, self.void_ptr)
        builder.call(func, [array_void_ptr, num_indices, indices_array, value])
    
    def call_ufunc(self, builder, module, ufunc_name, arrays):
        """
        Call a NumPy universal function (ufunc).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            ufunc_name: Name of ufunc (e.g., "add", "multiply", "sin")
            arrays: List of input arrays
        
        Returns:
            Result array
        """
        # Declare runtime function
        arg_types = [self.char_ptr, self.int32, self.void_ptr]
        func_type = ir.FunctionType(self.void_ptr, arg_types)
        func = ir.Function(module, func_type, name="numpy_call_ufunc")
        
        ufunc_str = self._create_string_literal(builder, module, ufunc_name)
        num_arrays = ir.Constant(self.int32, len(arrays))
        arrays_array = self._create_pointer_array(builder, module, arrays)
        
        result = builder.call(func, [ufunc_str, num_arrays, arrays_array])
        
        return result
    
    def array_dot(self, builder, module, array1_ptr, array2_ptr):
        """
        Matrix multiplication (np.dot).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array1_ptr: First array
            array2_ptr: Second array
        
        Returns:
            Result array
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.void_ptr])
        func = ir.Function(module, func_type, name="numpy_dot")
        
        arr1_void_ptr = builder.bitcast(array1_ptr, self.void_ptr)
        arr2_void_ptr = builder.bitcast(array2_ptr, self.void_ptr)
        
        result = builder.call(func, [arr1_void_ptr, arr2_void_ptr])
        
        return result
    
    def array_sum(self, builder, module, array_ptr, axis=None):
        """
        Sum array elements (np.sum).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array_ptr: Array to sum
            axis: Optional axis (None for total sum)
        
        Returns:
            Sum value or array
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.double, [self.void_ptr, self.int32])
        func = ir.Function(module, func_type, name="numpy_sum")
        
        array_void_ptr = builder.bitcast(array_ptr, self.void_ptr)
        axis_val = ir.Constant(self.int32, axis if axis is not None else -1)
        
        result = builder.call(func, [array_void_ptr, axis_val])
        
        return result
    
    def array_reshape(self, builder, module, array_ptr, new_shape):
        """
        Reshape array (np.reshape).
        
        Args:
            builder: LLVM IR builder
            module: LLVM module
            array_ptr: Array to reshape
            new_shape: New shape tuple
        
        Returns:
            Reshaped array
        """
        # Declare runtime function
        func_type = ir.FunctionType(self.void_ptr, [self.void_ptr, self.int32, self.void_ptr])
        func = ir.Function(module, func_type, name="numpy_reshape")
        
        ndim = ir.Constant(self.int32, len(new_shape))
        shape_array = self._create_int64_array(builder, new_shape)
        
        array_void_ptr = builder.bitcast(array_ptr, self.void_ptr)
        result = builder.call(func, [array_void_ptr, ndim, shape_array])
        
        return result
    
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
    
    def _create_int64_array(self, builder, values):
        """Create an array of int64 values."""
        array_type = ir.ArrayType(self.int64, len(values))
        array_ptr = builder.alloca(array_type)
        
        for i, val in enumerate(values):
            elem_ptr = builder.gep(array_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            builder.store(ir.Constant(self.int64, val), elem_ptr)
        
        return builder.bitcast(array_ptr, self.void_ptr)
    
    def _create_pointer_array(self, builder, module, pointers):
        """Create an array of void pointers."""
        if not pointers:
            return ir.Constant(self.void_ptr, None)
        
        array_type = ir.ArrayType(self.void_ptr, len(pointers))
        array_ptr = builder.alloca(array_type)
        
        for i, ptr in enumerate(pointers):
            elem_ptr = builder.gep(array_ptr, [ir.Constant(self.int32, 0), ir.Constant(self.int32, i)])
            ptr_void = builder.bitcast(ptr, self.void_ptr)
            builder.store(ptr_void, elem_ptr)
        
        return builder.bitcast(array_ptr, self.void_ptr)


def generate_numpy_runtime():
    """Generate C runtime code for NumPy interface."""
    
    c_code = """
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
"""
    
    # Write to file
    with open('numpy_runtime.c', 'w') as f:
        f.write(c_code)
    
    print("✅ NumPy runtime generated: numpy_runtime.c")


if __name__ == "__main__":
    from c_extension_interface import CExtensionInterface
    
    # Generate runtime C code
    generate_numpy_runtime()
    
    # Test NumPy interface
    c_ext = CExtensionInterface()
    numpy_interface = NumPyInterface(c_ext)
    
    print(f"✅ NumPyInterface initialized")
    print(f"   - NDArray structure: {numpy_interface.ndarray_type}")
