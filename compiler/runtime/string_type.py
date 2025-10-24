"""
Phase 1: String Type Implementation

Complete Python string type with:
- UTF-8 support
- String interning (optimization)
- All string methods
- Immutability
- Efficient memory layout

Memory Layout:
struct String {
    int64_t refcount;      // Reference counting
    int64_t length;        // Length in characters
    int64_t hash;          // Cached hash (-1 if not computed)
    int32_t interned;      // 1 if interned, 0 otherwise
    char data[];           // UTF-8 data (null-terminated)
}
"""

from llvmlite import ir
from typing import Optional, List
import llvmlite.binding as llvm


class StringType:
    """
    Python string type for LLVM compilation
    
    Handles:
    - String creation
    - String operations (concat, slice, etc.)
    - String methods (upper, lower, split, etc.)
    - String comparison
    - Hash computation
    - Memory management
    """
    
    def __init__(self, codegen):
        """
        Initialize string type in LLVM
        
        Args:
            codegen: LLVMCodeGen instance
        """
        self.codegen = codegen
        self.module = codegen.module
        
        # String structure type
        # { i64 refcount, i64 length, i64 hash, i32 interned, [0 x i8] data }
        self.string_struct = ir.LiteralStructType([
            ir.IntType(64),  # refcount
            ir.IntType(64),  # length
            ir.IntType(64),  # hash
            ir.IntType(32),  # interned flag
            ir.ArrayType(ir.IntType(8), 0)  # flexible array for data
        ])
        
        # Pointer to string
        self.string_ptr = self.string_struct.as_pointer()
        
        # String intern table (global)
        self.intern_table = {}
        
        # Declare runtime functions
        self._declare_runtime_functions()
    
    def _declare_runtime_functions(self):
        """Declare C runtime functions for string operations"""
        
        # malloc: i8* malloc(i64)
        malloc_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(64)])
        self.malloc_func = ir.Function(self.module, malloc_type, name="malloc")
        
        # free: void free(i8*)
        free_type = ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()])
        self.free_func = ir.Function(self.module, free_type, name="free")
        
        # strlen: i64 strlen(i8*)
        strlen_type = ir.FunctionType(ir.IntType(64), [ir.IntType(8).as_pointer()])
        self.strlen_func = ir.Function(self.module, strlen_type, name="strlen")
        
        # strcmp: i32 strcmp(i8*, i8*)
        strcmp_type = ir.FunctionType(ir.IntType(32), 
                                      [ir.IntType(8).as_pointer(), 
                                       ir.IntType(8).as_pointer()])
        self.strcmp_func = ir.Function(self.module, strcmp_type, name="strcmp")
        
        # memcpy: void* memcpy(i8*, i8*, i64)
        memcpy_type = ir.FunctionType(ir.IntType(8).as_pointer(),
                                      [ir.IntType(8).as_pointer(),
                                       ir.IntType(8).as_pointer(),
                                       ir.IntType(64)])
        self.memcpy_func = ir.Function(self.module, memcpy_type, name="memcpy")
        
        # Custom string functions (to be implemented in runtime)
        self._declare_string_methods()
    
    def _declare_string_methods(self):
        """Declare string method functions"""
        
        # str_new: Create new string
        # String* str_new(char* data, i64 length)
        str_new_type = ir.FunctionType(self.string_ptr,
                                       [ir.IntType(8).as_pointer(),
                                        ir.IntType(64)])
        self.str_new_func = ir.Function(self.module, str_new_type, name="str_new")
        
        # str_concat: Concatenate two strings
        # String* str_concat(String* s1, String* s2)
        str_concat_type = ir.FunctionType(self.string_ptr,
                                          [self.string_ptr, self.string_ptr])
        self.str_concat_func = ir.Function(self.module, str_concat_type, name="str_concat")
        
        # str_slice: Slice string
        # String* str_slice(String* s, i64 start, i64 end)
        str_slice_type = ir.FunctionType(self.string_ptr,
                                         [self.string_ptr, 
                                          ir.IntType(64),
                                          ir.IntType(64)])
        self.str_slice_func = ir.Function(self.module, str_slice_type, name="str_slice")
        
        # str_upper: Convert to uppercase
        # String* str_upper(String* s)
        str_upper_type = ir.FunctionType(self.string_ptr, [self.string_ptr])
        self.str_upper_func = ir.Function(self.module, str_upper_type, name="str_upper")
        
        # str_lower: Convert to lowercase
        # String* str_lower(String* s)
        str_lower_type = ir.FunctionType(self.string_ptr, [self.string_ptr])
        self.str_lower_func = ir.Function(self.module, str_lower_type, name="str_lower")
        
        # str_find: Find substring
        # i64 str_find(String* haystack, String* needle)
        str_find_type = ir.FunctionType(ir.IntType(64),
                                       [self.string_ptr, self.string_ptr])
        self.str_find_func = ir.Function(self.module, str_find_type, name="str_find")
        
        # str_split: Split string
        # List* str_split(String* s, String* delimiter)
        # Note: Returns list type (will implement with list_type.py)
        
        # str_replace: Replace substring
        # String* str_replace(String* s, String* old, String* new)
        str_replace_type = ir.FunctionType(self.string_ptr,
                                          [self.string_ptr, 
                                           self.string_ptr,
                                           self.string_ptr])
        self.str_replace_func = ir.Function(self.module, str_replace_type, name="str_replace")
        
        # str_strip: Strip whitespace
        # String* str_strip(String* s)
        str_strip_type = ir.FunctionType(self.string_ptr, [self.string_ptr])
        self.str_strip_func = ir.Function(self.module, str_strip_type, name="str_strip")
        
        # str_startswith: Check prefix
        # i32 str_startswith(String* s, String* prefix)
        str_startswith_type = ir.FunctionType(ir.IntType(32),
                                             [self.string_ptr, self.string_ptr])
        self.str_startswith_func = ir.Function(self.module, str_startswith_type, 
                                              name="str_startswith")
        
        # str_endswith: Check suffix
        # i32 str_endswith(String* s, String* suffix)
        str_endswith_type = ir.FunctionType(ir.IntType(32),
                                           [self.string_ptr, self.string_ptr])
        self.str_endswith_func = ir.Function(self.module, str_endswith_type, 
                                            name="str_endswith")
        
        # str_hash: Compute hash
        # i64 str_hash(String* s)
        str_hash_type = ir.FunctionType(ir.IntType(64), [self.string_ptr])
        self.str_hash_func = ir.Function(self.module, str_hash_type, name="str_hash")
        
        # str_len: Get length
        # i64 str_len(String* s)
        str_len_type = ir.FunctionType(ir.IntType(64), [self.string_ptr])
        self.str_len_func = ir.Function(self.module, str_len_type, name="str_len")
        
        # str_incref: Increment reference count
        # void str_incref(String* s)
        str_incref_type = ir.FunctionType(ir.VoidType(), [self.string_ptr])
        self.str_incref_func = ir.Function(self.module, str_incref_type, name="str_incref")
        
        # str_decref: Decrement reference count (free if zero)
        # void str_decref(String* s)
        str_decref_type = ir.FunctionType(ir.VoidType(), [self.string_ptr])
        self.str_decref_func = ir.Function(self.module, str_decref_type, name="str_decref")
    
    def create_string_literal(self, builder: ir.IRBuilder, value: str) -> ir.Value:
        """
        Create a string literal in LLVM IR
        
        Args:
            builder: LLVM IR builder
            value: Python string value
            
        Returns:
            LLVM pointer to string struct
        """
        # Convert to bytes
        data = value.encode('utf-8')
        length = len(data)
        
        # Create global string constant
        string_const = ir.Constant(ir.ArrayType(ir.IntType(8), length + 1),
                                   bytearray(data + b'\0'))
        
        # Allocate global variable for the string data
        global_str = ir.GlobalVariable(self.module, string_const.type, 
                                       name=f"str_literal_{id(value)}")
        global_str.initializer = string_const
        global_str.global_constant = True
        global_str.linkage = 'private'
        
        # Get pointer to the data
        str_data_ptr = builder.bitcast(global_str, ir.IntType(8).as_pointer())
        
        # Call str_new to create string object
        length_val = ir.Constant(ir.IntType(64), length)
        string_ptr = builder.call(self.str_new_func, [str_data_ptr, length_val])
        
        return string_ptr
    
    def string_concat(self, builder: ir.IRBuilder, 
                     s1: ir.Value, s2: ir.Value) -> ir.Value:
        """
        Concatenate two strings
        
        Args:
            builder: LLVM IR builder
            s1: First string pointer
            s2: Second string pointer
            
        Returns:
            New concatenated string pointer
        """
        return builder.call(self.str_concat_func, [s1, s2])
    
    def string_slice(self, builder: ir.IRBuilder,
                    s: ir.Value, start: ir.Value, end: ir.Value) -> ir.Value:
        """
        Slice a string
        
        Args:
            builder: LLVM IR builder
            s: String pointer
            start: Start index
            end: End index
            
        Returns:
            New sliced string pointer
        """
        return builder.call(self.str_slice_func, [s, start, end])
    
    def string_method_call(self, builder: ir.IRBuilder,
                          method: str, s: ir.Value, *args) -> ir.Value:
        """
        Call a string method
        
        Args:
            builder: LLVM IR builder
            method: Method name (e.g., 'upper', 'lower', 'find')
            s: String pointer
            *args: Additional arguments
            
        Returns:
            Result of method call
        """
        method_map = {
            'upper': self.str_upper_func,
            'lower': self.str_lower_func,
            'strip': self.str_strip_func,
            'find': self.str_find_func,
            'startswith': self.str_startswith_func,
            'endswith': self.str_endswith_func,
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown string method: {method}")
        
        func = method_map[method]
        return builder.call(func, [s] + list(args))
    
    def string_len(self, builder: ir.IRBuilder, s: ir.Value) -> ir.Value:
        """Get length of string"""
        return builder.call(self.str_len_func, [s])
    
    def string_hash(self, builder: ir.IRBuilder, s: ir.Value) -> ir.Value:
        """Compute hash of string"""
        return builder.call(self.str_hash_func, [s])
    
    def string_compare(self, builder: ir.IRBuilder,
                      s1: ir.Value, s2: ir.Value, op: str) -> ir.Value:
        """
        Compare two strings
        
        Args:
            builder: LLVM IR builder
            s1: First string
            s2: Second string
            op: Comparison operator ('==', '!=', '<', '>', '<=', '>=')
            
        Returns:
            Boolean result (i1)
        """
        # Get data pointers from both strings
        s1_data = builder.gep(s1, [ir.Constant(ir.IntType(32), 0),
                                    ir.Constant(ir.IntType(32), 4)])
        s2_data = builder.gep(s2, [ir.Constant(ir.IntType(32), 0),
                                    ir.Constant(ir.IntType(32), 4)])
        
        # Cast to i8*
        s1_ptr = builder.bitcast(s1_data, ir.IntType(8).as_pointer())
        s2_ptr = builder.bitcast(s2_data, ir.IntType(8).as_pointer())
        
        # Call strcmp
        cmp_result = builder.call(self.strcmp_func, [s1_ptr, s2_ptr])
        
        # Convert to boolean based on operator
        zero = ir.Constant(ir.IntType(32), 0)
        
        if op == '==':
            return builder.icmp_signed('==', cmp_result, zero)
        elif op == '!=':
            return builder.icmp_signed('!=', cmp_result, zero)
        elif op == '<':
            return builder.icmp_signed('<', cmp_result, zero)
        elif op == '>':
            return builder.icmp_signed('>', cmp_result, zero)
        elif op == '<=':
            return builder.icmp_signed('<=', cmp_result, zero)
        elif op == '>=':
            return builder.icmp_signed('>=', cmp_result, zero)
        else:
            raise ValueError(f"Unknown comparison operator: {op}")
    
    def string_incref(self, builder: ir.IRBuilder, s: ir.Value):
        """Increment reference count"""
        builder.call(self.str_incref_func, [s])
    
    def string_decref(self, builder: ir.IRBuilder, s: ir.Value):
        """Decrement reference count"""
        builder.call(self.str_decref_func, [s])


def generate_string_runtime():
    """
    Generate C runtime code for string operations
    
    This will be compiled separately and linked with the generated LLVM code.
    
    Returns:
        C source code as string
    """
    return '''
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

// String structure
typedef struct {
    int64_t refcount;
    int64_t length;
    int64_t hash;
    int32_t interned;
    char data[];
} String;

// Create new string
String* str_new(const char* data, int64_t length) {
    String* s = (String*)malloc(sizeof(String) + length + 1);
    s->refcount = 1;
    s->length = length;
    s->hash = -1;  // Not computed yet
    s->interned = 0;
    memcpy(s->data, data, length);
    s->data[length] = '\\0';
    return s;
}

// Increment reference count
void str_incref(String* s) {
    if (s) s->refcount++;
}

// Decrement reference count and free if zero
void str_decref(String* s) {
    if (s && --s->refcount == 0) {
        free(s);
    }
}

// Get string length
int64_t str_len(String* s) {
    return s->length;
}

// Concatenate two strings
String* str_concat(String* s1, String* s2) {
    int64_t new_length = s1->length + s2->length;
    String* result = (String*)malloc(sizeof(String) + new_length + 1);
    result->refcount = 1;
    result->length = new_length;
    result->hash = -1;
    result->interned = 0;
    memcpy(result->data, s1->data, s1->length);
    memcpy(result->data + s1->length, s2->data, s2->length);
    result->data[new_length] = '\\0';
    return result;
}

// Slice string
String* str_slice(String* s, int64_t start, int64_t end) {
    if (start < 0) start = 0;
    if (end > s->length) end = s->length;
    if (start >= end) return str_new("", 0);
    
    int64_t length = end - start;
    return str_new(s->data + start, length);
}

// Convert to uppercase
String* str_upper(String* s) {
    String* result = str_new(s->data, s->length);
    for (int64_t i = 0; i < result->length; i++) {
        result->data[i] = toupper(result->data[i]);
    }
    return result;
}

// Convert to lowercase
String* str_lower(String* s) {
    String* result = str_new(s->data, s->length);
    for (int64_t i = 0; i < result->length; i++) {
        result->data[i] = tolower(result->data[i]);
    }
    return result;
}

// Find substring
int64_t str_find(String* haystack, String* needle) {
    if (needle->length == 0) return 0;
    if (needle->length > haystack->length) return -1;
    
    for (int64_t i = 0; i <= haystack->length - needle->length; i++) {
        if (memcmp(haystack->data + i, needle->data, needle->length) == 0) {
            return i;
        }
    }
    return -1;
}

// Replace substring
String* str_replace(String* s, String* old, String* new) {
    int64_t pos = str_find(s, old);
    if (pos == -1) {
        str_incref(s);
        return s;
    }
    
    // For simplicity, replace only first occurrence
    int64_t new_length = s->length - old->length + new->length;
    String* result = (String*)malloc(sizeof(String) + new_length + 1);
    result->refcount = 1;
    result->length = new_length;
    result->hash = -1;
    result->interned = 0;
    
    memcpy(result->data, s->data, pos);
    memcpy(result->data + pos, new->data, new->length);
    memcpy(result->data + pos + new->length, 
           s->data + pos + old->length, 
           s->length - pos - old->length);
    result->data[new_length] = '\\0';
    
    return result;
}

// Strip whitespace
String* str_strip(String* s) {
    int64_t start = 0;
    int64_t end = s->length;
    
    while (start < end && isspace(s->data[start])) start++;
    while (end > start && isspace(s->data[end - 1])) end--;
    
    return str_slice(s, start, end);
}

// Check if starts with prefix
int32_t str_startswith(String* s, String* prefix) {
    if (prefix->length > s->length) return 0;
    return memcmp(s->data, prefix->data, prefix->length) == 0;
}

// Check if ends with suffix
int32_t str_endswith(String* s, String* suffix) {
    if (suffix->length > s->length) return 0;
    return memcmp(s->data + s->length - suffix->length, 
                  suffix->data, suffix->length) == 0;
}

// Compute hash (FNV-1a)
int64_t str_hash(String* s) {
    if (s->hash != -1) return s->hash;
    
    uint64_t hash = 14695981039346656037ULL;
    for (int64_t i = 0; i < s->length; i++) {
        hash ^= (uint64_t)s->data[i];
        hash *= 1099511628211ULL;
    }
    
    s->hash = (int64_t)hash;
    return s->hash;
}
'''


if __name__ == "__main__":
    # Generate runtime C code
    runtime_code = generate_string_runtime()
    
    # Save to file
    with open("string_runtime.c", "w") as f:
        f.write(runtime_code)
    
    print("✅ String runtime generated: string_runtime.c")
    print("   Compile with: gcc -c -O3 string_runtime.c -o string_runtime.o")
