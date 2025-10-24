"""
Symbol Table - Manage symbols and scoping

This module provides a comprehensive symbol table for tracking:
- Variable declarations and types
- Function signatures
- Scope hierarchies
- Symbol resolution

Phase: 1.1 (Frontend)
"""

from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum

from compiler.frontend.semantic import Type, TypeKind


class SymbolKind(Enum):
    """Kind of symbol"""
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FUNCTION = "function"
    GLOBAL = "global"


@dataclass
class Symbol:
    """Represents a symbol in the symbol table"""
    name: str
    kind: SymbolKind
    typ: Type
    scope_level: int = 0
    defined_at: int = 0  # Line number where defined
    used_at: List[int] = field(default_factory=list)  # Line numbers where used
    is_mutable: bool = True
    is_captured: bool = False  # For closures (Phase 2+)
    
    def __str__(self):
        return f"Symbol({self.name}: {self.typ}, {self.kind.value})"


@dataclass
class FunctionSignature:
    """Represents a function signature"""
    name: str
    param_types: List[Type]
    param_names: List[str]
    return_type: Type
    is_compiled: bool = False
    
    def __str__(self):
        params = ", ".join(f"{n}: {t}" for n, t in zip(self.param_names, self.param_types))
        return f"{self.name}({params}) -> {self.return_type}"


class SymbolTable:
    """
    Manages symbols in different scopes
    
    Features:
    - Hierarchical scope management
    - Symbol lookup with scope chain
    - Type tracking
    - Usage tracking for optimization
    """
    
    def __init__(self, parent: Optional['SymbolTable'] = None, name: str = "global"):
        self.parent = parent
        self.name = name
        self.symbols: Dict[str, Symbol] = {}
        self.functions: Dict[str, FunctionSignature] = {}
        self.children: List['SymbolTable'] = []
        
        # Calculate scope level
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1
            parent.children.append(self)
    
    def define_variable(
        self,
        name: str,
        typ: Type,
        kind: SymbolKind = SymbolKind.VARIABLE,
        line: int = 0
    ) -> Symbol:
        """
        Define a new variable in this scope
        
        Args:
            name: Variable name
            typ: Variable type
            kind: Symbol kind (variable, parameter, etc.)
            line: Line number where defined
            
        Returns:
            The Symbol object
        """
        symbol = Symbol(
            name=name,
            kind=kind,
            typ=typ,
            scope_level=self.level,
            defined_at=line
        )
        self.symbols[name] = symbol
        return symbol
    
    def define_function(
        self,
        name: str,
        param_types: List[Type],
        param_names: List[str],
        return_type: Type
    ) -> FunctionSignature:
        """
        Define a function in this scope
        
        Args:
            name: Function name
            param_types: List of parameter types
            param_names: List of parameter names
            return_type: Function return type
            
        Returns:
            FunctionSignature object
        """
        sig = FunctionSignature(
            name=name,
            param_types=param_types,
            param_names=param_names,
            return_type=return_type
        )
        self.functions[name] = sig
        return sig
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """
        Look up a symbol in this scope or parent scopes
        
        Args:
            name: Symbol name
            
        Returns:
            Symbol if found, None otherwise
        """
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def lookup_local(self, name: str) -> Optional[Symbol]:
        """
        Look up a symbol only in this scope (not parents)
        
        Args:
            name: Symbol name
            
        Returns:
            Symbol if found in this scope, None otherwise
        """
        return self.symbols.get(name)
    
    def lookup_function(self, name: str) -> Optional[FunctionSignature]:
        """
        Look up a function signature
        
        Args:
            name: Function name
            
        Returns:
            FunctionSignature if found, None otherwise
        """
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_function(name)
        return None
    
    def update_usage(self, name: str, line: int):
        """
        Record that a symbol was used at a given line
        
        Args:
            name: Symbol name
            line: Line number where used
        """
        symbol = self.lookup(name)
        if symbol:
            symbol.used_at.append(line)
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """
        Get all symbols in this scope and parent scopes
        
        Returns:
            Dictionary of all accessible symbols
        """
        all_symbols = {}
        if self.parent:
            all_symbols.update(self.parent.get_all_symbols())
        all_symbols.update(self.symbols)
        return all_symbols
    
    def get_unused_symbols(self) -> List[Symbol]:
        """
        Find symbols that were defined but never used
        
        Returns:
            List of unused symbols
        """
        unused = []
        for symbol in self.symbols.values():
            if not symbol.used_at and symbol.kind == SymbolKind.VARIABLE:
                unused.append(symbol)
        return unused
    
    def print_table(self, indent: int = 0):
        """
        Print symbol table for debugging
        
        Args:
            indent: Indentation level
        """
        prefix = "  " * indent
        print(f"{prefix}Scope: {self.name} (level {self.level})")
        print(f"{prefix}{'─' * 60}")
        
        if self.functions:
            print(f"{prefix}Functions:")
            for func in self.functions.values():
                print(f"{prefix}  {func}")
        
        if self.symbols:
            print(f"{prefix}Symbols:")
            for symbol in self.symbols.values():
                usage = f"{len(symbol.used_at)} uses" if symbol.used_at else "unused"
                print(f"{prefix}  {symbol.name}: {symbol.typ} [{symbol.kind.value}, {usage}]")
        
        if self.children:
            print(f"{prefix}Child Scopes:")
            for child in self.children:
                child.print_table(indent + 1)


class SymbolTableBuilder:
    """
    Builds symbol tables from Python AST
    
    This is a helper class that works with the semantic analyzer
    to build complete symbol tables with type information.
    """
    
    def __init__(self):
        self.global_table = SymbolTable(name="global")
        self.current_table = self.global_table
        
    def push_scope(self, name: str = "local") -> SymbolTable:
        """
        Create and enter a new scope
        
        Args:
            name: Name for the new scope
            
        Returns:
            The new SymbolTable
        """
        new_table = SymbolTable(parent=self.current_table, name=name)
        self.current_table = new_table
        return new_table
    
    def pop_scope(self):
        """Exit current scope and return to parent"""
        if self.current_table.parent:
            self.current_table = self.current_table.parent
    
    def define_variable(
        self,
        name: str,
        typ: Type,
        kind: SymbolKind = SymbolKind.VARIABLE,
        line: int = 0
    ) -> Symbol:
        """Define a variable in current scope"""
        return self.current_table.define_variable(name, typ, kind, line)
    
    def define_function(
        self,
        name: str,
        param_types: List[Type],
        param_names: List[str],
        return_type: Type
    ) -> FunctionSignature:
        """Define a function in current scope"""
        return self.current_table.define_function(
            name, param_types, param_names, return_type
        )
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol"""
        return self.current_table.lookup(name)
    
    def lookup_function(self, name: str) -> Optional[FunctionSignature]:
        """Look up a function"""
        return self.current_table.lookup_function(name)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("SYMBOL TABLE - Phase 1.1")
    print("=" * 80)
    
    # Create a global symbol table
    global_table = SymbolTable(name="global")
    
    # Define a function
    print("\n--- Defining function: add(x: int, y: int) -> int ---")
    func_sig = global_table.define_function(
        name="add",
        param_types=[Type(TypeKind.INT), Type(TypeKind.INT)],
        param_names=["x", "y"],
        return_type=Type(TypeKind.INT)
    )
    print(f"✅ Defined: {func_sig}")
    
    # Create function scope
    func_table = SymbolTable(parent=global_table, name="add")
    
    # Define parameters
    print("\n--- Defining parameters in function scope ---")
    func_table.define_variable("x", Type(TypeKind.INT), SymbolKind.PARAMETER, line=1)
    func_table.define_variable("y", Type(TypeKind.INT), SymbolKind.PARAMETER, line=1)
    print("✅ Parameters defined")
    
    # Define local variable
    print("\n--- Defining local variable: result ---")
    result_sym = func_table.define_variable("result", Type(TypeKind.INT), line=2)
    print(f"✅ Defined: {result_sym}")
    
    # Record usage
    print("\n--- Recording variable usage ---")
    func_table.update_usage("x", line=2)
    func_table.update_usage("y", line=2)
    func_table.update_usage("result", line=3)
    print("✅ Usage recorded")
    
    # Lookup tests
    print("\n--- Symbol lookup tests ---")
    x_sym = func_table.lookup("x")
    print(f"Lookup 'x': {x_sym}")
    
    unknown = func_table.lookup("unknown")
    print(f"Lookup 'unknown': {unknown}")
    
    # Find unused symbols
    print("\n--- Checking for unused symbols ---")
    unused = func_table.get_unused_symbols()
    if unused:
        print(f"⚠️  Unused symbols: {[s.name for s in unused]}")
    else:
        print("✅ No unused symbols")
    
    # Print complete symbol table
    print("\n--- Complete Symbol Table ---")
    global_table.print_table()
    
    # Test with nested scopes
    print("\n" + "=" * 80)
    print("--- Testing Nested Scopes ---")
    print("=" * 80)
    
    builder = SymbolTableBuilder()
    
    # Define function in global scope
    builder.define_function(
        "outer",
        [Type(TypeKind.INT)],
        ["n"],
        Type(TypeKind.INT)
    )
    
    # Enter function scope
    builder.push_scope("outer")
    builder.define_variable("n", Type(TypeKind.INT), SymbolKind.PARAMETER)
    builder.define_variable("total", Type(TypeKind.INT))
    
    # Enter nested scope (loop)
    builder.push_scope("for_loop")
    builder.define_variable("i", Type(TypeKind.INT))
    
    # Lookup should find variables from outer scopes
    n_sym = builder.lookup("n")
    total_sym = builder.lookup("total")
    i_sym = builder.lookup("i")
    
    print(f"From inner scope, found:")
    print(f"  n: {n_sym}")
    print(f"  total: {total_sym}")
    print(f"  i: {i_sym}")
    
    # Pop back to function scope
    builder.pop_scope()
    
    # i should not be visible anymore
    i_after_pop = builder.lookup("i")
    print(f"\nAfter popping scope, 'i' lookup: {i_after_pop}")
    
    # Print final table
    print("\n--- Final Symbol Table ---")
    builder.global_table.print_table()
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.1 Symbol Table Complete!")
    print("=" * 80)
    print("\n🎉 Phase 1.1 (Frontend) COMPLETE!")
    print("Next: Phase 1.2 - Intermediate Representation (IR)")
