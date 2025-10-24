"""
Phase 3.1: List Support - Implementation

Adds support for Python lists in the compiler:
- List literals: [1, 2, 3]
- List indexing: lst[0]
- List operations: append, len, iteration
- Type-specialized list implementations

This is the first step in Phase 3.1: Expanded Language Support
"""

import ast
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ListType:
    """Represents a homogeneous list type"""
    element_type: str  # 'int', 'float', 'bool', etc.
    
    def __str__(self):
        return f"List[{self.element_type}]"


class ListOperations:
    """
    Defines list operations that can be compiled to native code.
    
    Strategy:
    - For homogeneous lists with known element type, generate specialized code
    - For dynamic lists, fallback to runtime support
    """
    
    SUPPORTED_OPERATIONS = [
        'append',
        'len',
        'index',
        'clear',
        'copy',
        'count',
        'extend',
        'insert',
        'pop',
        'remove',
        'reverse',
        'sort'
    ]
    
    @staticmethod
    def is_list_operation(node: ast.AST) -> bool:
        """Check if AST node is a list operation"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                return node.func.attr in ListOperations.SUPPORTED_OPERATIONS
        return False
    
    @staticmethod
    def infer_list_type(elements: List[ast.AST]) -> Optional[ListType]:
        """
        Infer the element type of a list from its literal elements.
        Returns None if type cannot be determined or is heterogeneous.
        """
        if not elements:
            return None  # Empty list, cannot infer
        
        # Check if all elements have compatible types
        element_types = set()
        for elem in elements:
            if isinstance(elem, ast.Constant):
                if isinstance(elem.value, int):
                    element_types.add('int')
                elif isinstance(elem.value, float):
                    element_types.add('float')
                elif isinstance(elem.value, bool):
                    element_types.add('bool')
                elif isinstance(elem.value, str):
                    element_types.add('str')
        
        if len(element_types) == 1:
            return ListType(element_type=element_types.pop())
        elif len(element_types) == 2 and 'int' in element_types and 'float' in element_types:
            # int and float can be promoted to float
            return ListType(element_type='float')
        else:
            return None  # Heterogeneous or unknown


class ListLowering:
    """
    Lowers Python list operations to IR operations.
    
    Two strategies:
    1. Specialized lists (List[int], List[float]): Direct memory operations
    2. Dynamic lists: Use runtime library with type tagging
    """
    
    def __init__(self):
        self.list_counter = 0
    
    def lower_list_literal(self, node: ast.List, inferred_type: Optional[ListType]) -> str:
        """
        Lower a list literal to IR code.
        
        For specialized lists:
            %list = alloc_list_int(3)  # capacity 3
            store_list_int(%list, 0, 1)
            store_list_int(%list, 1, 2)
            store_list_int(%list, 2, 3)
        
        For dynamic lists:
            %list = alloc_list_dynamic(3)
            store_list_dynamic(%list, 0, box_int(1))
            store_list_dynamic(%list, 1, box_int(2))
            store_list_dynamic(%list, 2, box_int(3))
        """
        list_id = self.list_counter
        self.list_counter += 1
        
        ir_code = []
        list_var = f"%list{list_id}"
        
        if inferred_type:
            # Specialized list
            elem_type = inferred_type.element_type
            capacity = len(node.elts)
            
            ir_code.append(f"{list_var} = alloc_list_{elem_type}({capacity})")
            
            for i, elem in enumerate(node.elts):
                elem_ir = self._lower_expr(elem)
                ir_code.append(f"store_list_{elem_type}({list_var}, {i}, {elem_ir})")
        else:
            # Dynamic list
            capacity = len(node.elts)
            ir_code.append(f"{list_var} = alloc_list_dynamic({capacity})")
            
            for i, elem in enumerate(node.elts):
                elem_ir = self._lower_expr(elem)
                ir_code.append(f"store_list_dynamic({list_var}, {i}, box_value({elem_ir}))")
        
        return "\n".join(ir_code), list_var
    
    def lower_list_index(self, list_var: str, index_expr: ast.AST, 
                        list_type: Optional[ListType]) -> str:
        """
        Lower list indexing: lst[index]
        
        Specialized: load_list_int(%list, %index) -> int
        Dynamic: unbox(load_list_dynamic(%list, %index))
        """
        index_ir = self._lower_expr(index_expr)
        
        if list_type:
            elem_type = list_type.element_type
            return f"load_list_{elem_type}({list_var}, {index_ir})"
        else:
            return f"load_list_dynamic({list_var}, {index_ir})"
    
    def lower_list_append(self, list_var: str, value_expr: ast.AST,
                         list_type: Optional[ListType]) -> str:
        """
        Lower list.append(value)
        
        Specialized: append_list_int(%list, %value)
        Dynamic: append_list_dynamic(%list, box_value(%value))
        """
        value_ir = self._lower_expr(value_expr)
        
        if list_type:
            elem_type = list_type.element_type
            return f"append_list_{elem_type}({list_var}, {value_ir})"
        else:
            return f"append_list_dynamic({list_var}, box_value({value_ir}))"
    
    def lower_list_len(self, list_var: str) -> str:
        """Lower len(lst) -> int"""
        return f"list_len({list_var})"
    
    def _lower_expr(self, expr: ast.AST) -> str:
        """Helper to lower expression to IR (simplified)"""
        if isinstance(expr, ast.Constant):
            return str(expr.value)
        elif isinstance(expr, ast.Name):
            return f"%{expr.id}"
        else:
            return "%temp"


def demo():
    """Demonstrate list support design"""
    print("\n" + "="*70)
    print("PHASE 3.1: LIST SUPPORT - DESIGN DEMO")
    print("="*70 + "\n")
    
    # Example 1: Homogeneous integer list
    code1 = "[1, 2, 3]"
    tree1 = ast.parse(code1, mode='eval')
    list_node1 = tree1.body
    
    if isinstance(list_node1, ast.List):
        list_type1 = ListOperations.infer_list_type(list_node1.elts)
        print(f"Example 1: {code1}")
        print(f"  Inferred Type: {list_type1}")
        print(f"  Strategy: Specialized List[int] with direct memory access")
        print()
    
    # Example 2: Mixed int/float list
    code2 = "[1, 2.5, 3]"
    tree2 = ast.parse(code2, mode='eval')
    list_node2 = tree2.body
    
    if isinstance(list_node2, ast.List):
        list_type2 = ListOperations.infer_list_type(list_node2.elts)
        print(f"Example 2: {code2}")
        print(f"  Inferred Type: {list_type2}")
        print(f"  Strategy: Specialized List[float] with type promotion")
        print()
    
    # Example 3: IR generation
    print("Example 3: IR Generation for [1, 2, 3]")
    print("─" * 70)
    lowering = ListLowering()
    ir_code, list_var = lowering.lower_list_literal(list_node1, list_type1)
    print(ir_code)
    print(f"\nResult variable: {list_var}")
    print()
    
    # Example 4: List operations
    print("Example 4: Supported Operations")
    print("─" * 70)
    for op in ListOperations.SUPPORTED_OPERATIONS:
        print(f"  • {op}()")
    print()
    
    # Example 5: Runtime library functions needed
    print("Example 5: Runtime Library Functions (to be implemented)")
    print("─" * 70)
    runtime_funcs = [
        "alloc_list_int(capacity: int) -> List*",
        "alloc_list_float(capacity: int) -> List*",
        "store_list_int(list: List*, index: int, value: int)",
        "load_list_int(list: List*, index: int) -> int",
        "append_list_int(list: List*, value: int)",
        "list_len(list: List*) -> int",
        "free_list(list: List*)",
    ]
    for func in runtime_funcs:
        print(f"  • {func}")
    print()
    
    print("="*70)
    print("Next Steps:")
    print("  1. Implement runtime library for list operations")
    print("  2. Integrate list support into IR")
    print("  3. Add LLVM code generation for lists")
    print("  4. Create test suite for list operations")
    print("  5. Benchmark specialized vs dynamic lists")
    print("="*70)


if __name__ == "__main__":
    demo()
