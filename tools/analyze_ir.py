"""
Test IR optimization - analyze generated IR for redundancies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.frontend.parser import Parser
from compiler.frontend.semantic import analyze
from compiler.ir.lowering import IRLowering

code = """
def compute(x: int) -> int:
    a: int = x + 5
    b: int = a * 2
    c: int = b - 3
    return c

def main() -> int:
    return compute(10)
"""

# Parse
parser = Parser()
parse_result = parser.parse_source(code)

# Build symbol table
from compiler.frontend.symbols import SymbolTableBuilder
symbol_builder = SymbolTableBuilder()
symbol_table = symbol_builder.build(parse_result.ast_tree)

# Semantic analysis
semantic_result = analyze(parse_result.ast_tree)

# Lower to IR
lowering = IRLowering(symbol_table)
ir_module = lowering.visit_Module(parse_result.ast_tree)

# Print IR
print("="*80)
print("GENERATED IR")
print("="*80)
print(ir_module)
print("="*80)
print("\nAnalysis:")
print(f"- Functions: {len(ir_module.functions)}")
for func in ir_module.functions:
    print(f"- {func.name}: {len(func.blocks)} blocks, ~{sum(len(b.instructions) for b in func.blocks)} instructions")
