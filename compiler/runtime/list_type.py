"""
Phase 1: List Type Implementation

Python list - dynamic array with:
- Resizable capacity
- Slicing support
- List methods (append, extend, pop, insert, etc.)
- Reference counting for elements
"""

from llvmlite import ir
from typing import Optional
import llvmlite.binding as llvm


class ListType:
    """Python list type for LLVM compilation"""
    
    def __init__(self, codegen):
        self.codegen = codegen
        self.module = codegen.module
        
        # List structure: { i64 refcount, i64 length, i64 capacity, void** items }
        self.list_struct = ir.LiteralStructType([
            ir.IntType(64),  # refcount
            ir.IntType(64),  # length (number of elements)
            ir.IntType(64),  # capacity (allocated space)
            ir.IntType(8).as_pointer().as_pointer()  # items array (void**)
        ])
        
        self.list_ptr = self.list_struct.as_pointer()
        self._declare_runtime_functions()
    
    def _declare_runtime_functions(self):
        """Declare list runtime functions"""
        
        # list_new: Create new empty list
        list_new_type = ir.FunctionType(self.list_ptr, [])
        self.list_new_func = ir.Function(self.module, list_new_type, name="list_new")
        
        # list_append: Append item
        list_append_type = ir.FunctionType(ir.VoidType(),
                                          [self.list_ptr, 
                                           ir.IntType(8).as_pointer()])
        self.list_append_func = ir.Function(self.module, list_append_type, name="list_append")
        
        # list_get: Get item at index
        list_get_type = ir.FunctionType(ir.IntType(8).as_pointer(),
                                       [self.list_ptr, ir.IntType(64)])
        self.list_get_func = ir.Function(self.module, list_get_type, name="list_get")
        
        # list_set: Set item at index
        list_set_type = ir.FunctionType(ir.VoidType(),
                                       [self.list_ptr, 
                                        ir.IntType(64),
                                        ir.IntType(8).as_pointer()])
        self.list_set_func = ir.Function(self.module, list_set_type, name="list_set")
        
        # list_len: Get length
        list_len_type = ir.FunctionType(ir.IntType(64), [self.list_ptr])
        self.list_len_func = ir.Function(self.module, list_len_type, name="list_len")
        
        # list_pop: Remove and return last item
        list_pop_type = ir.FunctionType(ir.IntType(8).as_pointer(), [self.list_ptr])
        self.list_pop_func = ir.Function(self.module, list_pop_type, name="list_pop")
        
        # list_slice: Slice list
        list_slice_type = ir.FunctionType(self.list_ptr,
                                         [self.list_ptr, 
                                          ir.IntType(64),
                                          ir.IntType(64)])
        self.list_slice_func = ir.Function(self.module, list_slice_type, name="list_slice")
        
        # list_incref/decref
        list_incref_type = ir.FunctionType(ir.VoidType(), [self.list_ptr])
        self.list_incref_func = ir.Function(self.module, list_incref_type, name="list_incref")
        
        list_decref_type = ir.FunctionType(ir.VoidType(), [self.list_ptr])
        self.list_decref_func = ir.Function(self.module, list_decref_type, name="list_decref")
    
    def create_list(self, builder: ir.IRBuilder) -> ir.Value:
        """Create new empty list"""
        return builder.call(self.list_new_func, [])
    
    def list_append(self, builder: ir.IRBuilder, lst: ir.Value, item: ir.Value):
        """Append item to list"""
        item_ptr = builder.bitcast(item, ir.IntType(8).as_pointer())
        builder.call(self.list_append_func, [lst, item_ptr])
    
    def list_get(self, builder: ir.IRBuilder, lst: ir.Value, index: ir.Value) -> ir.Value:
        """Get item at index"""
        return builder.call(self.list_get_func, [lst, index])
    
    def list_len(self, builder: ir.IRBuilder, lst: ir.Value) -> ir.Value:
        """Get list length"""
        return builder.call(self.list_len_func, [lst])


def generate_list_runtime():
    """Generate C runtime for list operations"""
    return '''
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int64_t refcount;
    int64_t length;
    int64_t capacity;
    void** items;
} List;

List* list_new() {
    List* lst = (List*)malloc(sizeof(List));
    lst->refcount = 1;
    lst->length = 0;
    lst->capacity = 8;
    lst->items = (void**)malloc(sizeof(void*) * lst->capacity);
    return lst;
}

void list_incref(List* lst) {
    if (lst) lst->refcount++;
}

void list_decref(List* lst) {
    if (lst && --lst->refcount == 0) {
        free(lst->items);
        free(lst);
    }
}

int64_t list_len(List* lst) {
    return lst->length;
}

void list_resize(List* lst) {
    if (lst->length >= lst->capacity) {
        lst->capacity *= 2;
        lst->items = (void**)realloc(lst->items, sizeof(void*) * lst->capacity);
    }
}

void list_append(List* lst, void* item) {
    list_resize(lst);
    lst->items[lst->length++] = item;
}

void* list_get(List* lst, int64_t index) {
    if (index < 0 || index >= lst->length) return NULL;
    return lst->items[index];
}

void list_set(List* lst, int64_t index, void* item) {
    if (index >= 0 && index < lst->length) {
        lst->items[index] = item;
    }
}

void* list_pop(List* lst) {
    if (lst->length == 0) return NULL;
    return lst->items[--lst->length];
}

List* list_slice(List* lst, int64_t start, int64_t end) {
    if (start < 0) start = 0;
    if (end > lst->length) end = lst->length;
    if (start >= end) return list_new();
    
    List* result = list_new();
    for (int64_t i = start; i < end; i++) {
        list_append(result, lst->items[i]);
    }
    return result;
}
'''


if __name__ == "__main__":
    with open("list_runtime.c", "w") as f:
        f.write(generate_list_runtime())
    print("✅ List runtime generated: list_runtime.c")
