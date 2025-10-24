"""
Phase 1: Tuple and Bool Types Implementation
"""

from llvmlite import ir
import llvmlite.binding as llvm


class TupleType:
    """Python tuple type (immutable sequence)"""
    
    def __init__(self, codegen):
        self.codegen = codegen
        self.module = codegen.module
        
        # Tuple structure: { i64 refcount, i64 length, void** items }
        self.tuple_struct = ir.LiteralStructType([
            ir.IntType(64),  # refcount
            ir.IntType(64),  # length
            ir.IntType(8).as_pointer().as_pointer()  # items
        ])
        
        self.tuple_ptr = self.tuple_struct.as_pointer()
        self._declare_runtime_functions()
    
    def _declare_runtime_functions(self):
        """Declare tuple runtime functions"""
        
        # tuple_new: Create tuple with items
        tuple_new_type = ir.FunctionType(self.tuple_ptr,
                                        [ir.IntType(8).as_pointer().as_pointer(),
                                         ir.IntType(64)])
        self.tuple_new_func = ir.Function(self.module, tuple_new_type, name="tuple_new")
        
        # tuple_get: Get item at index
        tuple_get_type = ir.FunctionType(ir.IntType(8).as_pointer(),
                                        [self.tuple_ptr, ir.IntType(64)])
        self.tuple_get_func = ir.Function(self.module, tuple_get_type, name="tuple_get")
        
        # tuple_len: Get length
        tuple_len_type = ir.FunctionType(ir.IntType(64), [self.tuple_ptr])
        self.tuple_len_func = ir.Function(self.module, tuple_len_type, name="tuple_len")
        
        # tuple_incref/decref
        tuple_incref_type = ir.FunctionType(ir.VoidType(), [self.tuple_ptr])
        self.tuple_incref_func = ir.Function(self.module, tuple_incref_type, name="tuple_incref")
        
        tuple_decref_type = ir.FunctionType(ir.VoidType(), [self.tuple_ptr])
        self.tuple_decref_func = ir.Function(self.module, tuple_decref_type, name="tuple_decref")


class BoolType:
    """Python bool type (True/False)"""
    
    def __init__(self, codegen):
        self.codegen = codegen
        self.module = codegen.module
        
        # Bool is just an i1 in LLVM, but we need Python-compatible representation
        self.bool_type = ir.IntType(1)
        
        # Create global True and False constants
        self.true_val = ir.Constant(self.bool_type, 1)
        self.false_val = ir.Constant(self.bool_type, 0)
    
    def create_bool(self, builder: ir.IRBuilder, value: bool) -> ir.Value:
        """Create bool constant"""
        return self.true_val if value else self.false_val
    
    def bool_to_int(self, builder: ir.IRBuilder, b: ir.Value) -> ir.Value:
        """Convert bool to int"""
        return builder.zext(b, ir.IntType(64))
    
    def int_to_bool(self, builder: ir.IRBuilder, i: ir.Value) -> ir.Value:
        """Convert int to bool"""
        zero = ir.Constant(i.type, 0)
        return builder.icmp_signed('!=', i, zero)


class NoneType:
    """Python None type"""
    
    def __init__(self, codegen):
        self.codegen = codegen
        self.module = codegen.module
        
        # None is represented as a null pointer
        self.none_type = ir.IntType(8).as_pointer()
        self.none_val = ir.Constant(self.none_type, None)
    
    def create_none(self, builder: ir.IRBuilder) -> ir.Value:
        """Create None value"""
        return self.none_val
    
    def is_none(self, builder: ir.IRBuilder, value: ir.Value) -> ir.Value:
        """Check if value is None"""
        return builder.icmp_signed('==', value, self.none_val)


def generate_tuple_runtime():
    """Generate C runtime for tuple operations"""
    return '''
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int64_t refcount;
    int64_t length;
    void** items;
} Tuple;

Tuple* tuple_new(void** items, int64_t length) {
    Tuple* tup = (Tuple*)malloc(sizeof(Tuple));
    tup->refcount = 1;
    tup->length = length;
    tup->items = (void**)malloc(sizeof(void*) * length);
    memcpy(tup->items, items, sizeof(void*) * length);
    return tup;
}

void tuple_incref(Tuple* tup) {
    if (tup) tup->refcount++;
}

void tuple_decref(Tuple* tup) {
    if (tup && --tup->refcount == 0) {
        free(tup->items);
        free(tup);
    }
}

int64_t tuple_len(Tuple* tup) {
    return tup->length;
}

void* tuple_get(Tuple* tup, int64_t index) {
    if (index < 0 || index >= tup->length) return NULL;
    return tup->items[index];
}
'''


if __name__ == "__main__":
    with open("tuple_runtime.c", "w") as f:
        f.write(generate_tuple_runtime())
    print("✅ Tuple runtime generated: tuple_runtime.c")
