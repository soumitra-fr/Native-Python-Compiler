"""
Phase 1: Dict Type Implementation

Python dict - hash table with:
- Open addressing
- Key/value pairs
- Hash collision handling
- Dynamic resizing
"""

from llvmlite import ir
import llvmlite.binding as llvm


class DictType:
    """Python dict type for LLVM compilation"""
    
    def __init__(self, codegen):
        self.codegen = codegen
        self.module = codegen.module
        
        # Dict structure: { i64 refcount, i64 size, i64 capacity, Entry* entries }
        # Entry: { void* key, void* value, i64 hash, i32 occupied }
        self.dict_struct = ir.LiteralStructType([
            ir.IntType(64),  # refcount
            ir.IntType(64),  # size (number of items)
            ir.IntType(64),  # capacity
            ir.IntType(8).as_pointer()  # entries array
        ])
        
        self.dict_ptr = self.dict_struct.as_pointer()
        self._declare_runtime_functions()
    
    def _declare_runtime_functions(self):
        """Declare dict runtime functions"""
        
        # dict_new
        dict_new_type = ir.FunctionType(self.dict_ptr, [])
        self.dict_new_func = ir.Function(self.module, dict_new_type, name="dict_new")
        
        # dict_set
        dict_set_type = ir.FunctionType(ir.VoidType(),
                                       [self.dict_ptr,
                                        ir.IntType(8).as_pointer(),  # key
                                        ir.IntType(8).as_pointer()])  # value
        self.dict_set_func = ir.Function(self.module, dict_set_type, name="dict_set")
        
        # dict_get
        dict_get_type = ir.FunctionType(ir.IntType(8).as_pointer(),
                                       [self.dict_ptr,
                                        ir.IntType(8).as_pointer()])
        self.dict_get_func = ir.Function(self.module, dict_get_type, name="dict_get")
        
        # dict_len
        dict_len_type = ir.FunctionType(ir.IntType(64), [self.dict_ptr])
        self.dict_len_func = ir.Function(self.module, dict_len_type, name="dict_len")
        
        # dict_contains
        dict_contains_type = ir.FunctionType(ir.IntType(32),
                                            [self.dict_ptr,
                                             ir.IntType(8).as_pointer()])
        self.dict_contains_func = ir.Function(self.module, dict_contains_type, name="dict_contains")
        
        # dict_incref/decref
        dict_incref_type = ir.FunctionType(ir.VoidType(), [self.dict_ptr])
        self.dict_incref_func = ir.Function(self.module, dict_incref_type, name="dict_incref")
        
        dict_decref_type = ir.FunctionType(ir.VoidType(), [self.dict_ptr])
        self.dict_decref_func = ir.Function(self.module, dict_decref_type, name="dict_decref")
    
    def create_dict(self, builder: ir.IRBuilder) -> ir.Value:
        """Create new empty dict"""
        return builder.call(self.dict_new_func, [])
    
    def dict_set(self, builder: ir.IRBuilder, dct: ir.Value, 
                 key: ir.Value, value: ir.Value):
        """Set key-value pair"""
        key_ptr = builder.bitcast(key, ir.IntType(8).as_pointer())
        value_ptr = builder.bitcast(value, ir.IntType(8).as_pointer())
        builder.call(self.dict_set_func, [dct, key_ptr, value_ptr])
    
    def dict_get(self, builder: ir.IRBuilder, dct: ir.Value, key: ir.Value) -> ir.Value:
        """Get value for key"""
        key_ptr = builder.bitcast(key, ir.IntType(8).as_pointer())
        return builder.call(self.dict_get_func, [dct, key_ptr])


def generate_dict_runtime():
    """Generate C runtime for dict operations"""
    return '''
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    void* key;
    void* value;
    int64_t hash;
    int32_t occupied;
} DictEntry;

typedef struct {
    int64_t refcount;
    int64_t size;
    int64_t capacity;
    DictEntry* entries;
} Dict;

int64_t dict_hash_ptr(void* ptr) {
    return (int64_t)ptr;
}

Dict* dict_new() {
    Dict* dct = (Dict*)malloc(sizeof(Dict));
    dct->refcount = 1;
    dct->size = 0;
    dct->capacity = 16;
    dct->entries = (DictEntry*)calloc(dct->capacity, sizeof(DictEntry));
    return dct;
}

void dict_incref(Dict* dct) {
    if (dct) dct->refcount++;
}

void dict_decref(Dict* dct) {
    if (dct && --dct->refcount == 0) {
        free(dct->entries);
        free(dct);
    }
}

int64_t dict_len(Dict* dct) {
    return dct->size;
}

void dict_resize(Dict* dct) {
    if (dct->size >= dct->capacity * 0.75) {
        int64_t old_capacity = dct->capacity;
        DictEntry* old_entries = dct->entries;
        
        dct->capacity *= 2;
        dct->entries = (DictEntry*)calloc(dct->capacity, sizeof(DictEntry));
        dct->size = 0;
        
        for (int64_t i = 0; i < old_capacity; i++) {
            if (old_entries[i].occupied) {
                dict_set(dct, old_entries[i].key, old_entries[i].value);
            }
        }
        free(old_entries);
    }
}

void dict_set(Dict* dct, void* key, void* value) {
    dict_resize(dct);
    
    int64_t hash = dict_hash_ptr(key);
    int64_t index = hash % dct->capacity;
    
    while (dct->entries[index].occupied && dct->entries[index].key != key) {
        index = (index + 1) % dct->capacity;
    }
    
    if (!dct->entries[index].occupied) {
        dct->size++;
    }
    
    dct->entries[index].key = key;
    dct->entries[index].value = value;
    dct->entries[index].hash = hash;
    dct->entries[index].occupied = 1;
}

void* dict_get(Dict* dct, void* key) {
    int64_t hash = dict_hash_ptr(key);
    int64_t index = hash % dct->capacity;
    
    while (dct->entries[index].occupied) {
        if (dct->entries[index].key == key) {
            return dct->entries[index].value;
        }
        index = (index + 1) % dct->capacity;
    }
    return NULL;
}

int32_t dict_contains(Dict* dct, void* key) {
    return dict_get(dct, key) != NULL;
}
'''


if __name__ == "__main__":
    with open("dict_runtime.c", "w") as f:
        f.write(generate_dict_runtime())
    print("✅ Dict runtime generated: dict_runtime.c")
