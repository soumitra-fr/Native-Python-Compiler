"""
AST to IR Lowering - Convert Python AST to typed IR

This module converts Python AST nodes to our custom typed IR.
It performs:
- Type-directed lowering
- SSA construction
- Control flow graph building
- Temporary variable generation

Phase: 1.2 (IR)
"""

import ast
from typing import Optional, List, Dict
from dataclasses import dataclass

from compiler.frontend.semantic import Type, TypeKind, SemanticAnalyzer
from compiler.frontend.symbols import SymbolTable, SymbolTableBuilder
from compiler.ir.ir_nodes import *


class IRLowering(ast.NodeVisitor):
    """
    Lowers Python AST to typed IR
    
    Converts high-level Python constructs to low-level typed IR
    suitable for LLVM code generation.
    """
    
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.current_function: Optional[IRFunction] = None
        self.current_block: Optional[IRBasicBlock] = None
        self.module = IRModule("main")
        
        # State tracking
        self.temp_counter = 0
        self.label_counter = 0
        self.type_map: Dict[ast.AST, Type] = {}
        
        # Loop context for break/continue
        self.loop_stack: List[tuple] = []  # (break_label, continue_label)
    
    def new_temp(self, typ: Type) -> IRVar:
        """Generate a new temporary variable"""
        name = f"t{self.temp_counter}"
        self.temp_counter += 1
        return IRVar(name, typ)
    
    def new_label(self, prefix: str = "L") -> str:
        """Generate a new label name"""
        name = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return name
    
    def emit(self, instr: IRNode):
        """Emit instruction to current block"""
        if self.current_block:
            self.current_block.add_instruction(instr)
    
    def get_type(self, node: ast.AST) -> Type:
        """Get type of AST node from type map"""
        if node in self.type_map:
            return self.type_map[node]
        return Type(TypeKind.UNKNOWN)
    
    def visit_Module(self, node: ast.Module):
        """Lower module"""
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.visit_FunctionDef(stmt)
            elif isinstance(stmt, ast.AsyncFunctionDef):
                self.visit_AsyncFunctionDef(stmt)
            elif isinstance(stmt, ast.ClassDef):
                self.visit_ClassDef(stmt)
        return self.module
    
    def _contains_yield(self, node: ast.FunctionDef) -> bool:
        """Check if function contains yield or yield from"""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _lower_generator_function(self, node: ast.FunctionDef):
        """Lower generator function to state machine using coroutines"""
        # Extract signature
        param_nodes = []
        for arg in node.args.args:
            if arg.annotation:
                typ = self.ast_to_type(arg.annotation)
            else:
                typ = Type(TypeKind.UNKNOWN)
            param_nodes.append(IRVar(arg.arg, typ))
        
        # Return type for generator (yields return type)
        if node.returns:
            return_type = self.ast_to_type(node.returns)
        else:
            return_type = Type(TypeKind.UNKNOWN)
        
        # Create IR async function (generators use same infrastructure as async)
        # Note: We pass empty body list initially
        func = IRAsyncFunction(
            name=node.name,
            params=param_nodes,
            body=[],
            return_type=return_type,
            line=node.lineno
        )
        
        self.current_function = func
        self.temp_counter = 0
        self.label_counter = 0
        
        # Create entry block
        entry = IRBasicBlock("entry")
        self.current_block = entry
        func.add_block(entry)  # Now using add_block method
        
        # Lower function body (yields will be emitted as IRYield nodes)
        for stmt in node.body:
            self.visit(stmt)
        
        # Add implicit return if needed
        if self.current_block and self.current_block.instructions:
            last_instr = self.current_block.instructions[-1]
            if not isinstance(last_instr, (IRReturn, IRJump, IRBranch)):
                self.emit(IRReturn(None))
        elif self.current_block:
            self.emit(IRReturn(None))
        
        self.module.add_function(func)
        self.current_function = None
        self.current_block = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Lower function definition"""
        # Check if this is a generator function
        if self._contains_yield(node):
            return self._lower_generator_function(node)
        
        # Extract signature
        param_names = [arg.arg for arg in node.args.args]
        param_types = []
        
        for arg in node.args.args:
            if arg.annotation:
                typ = self.ast_to_type(arg.annotation)
            else:
                typ = Type(TypeKind.UNKNOWN)
            param_types.append(typ)
        
        # Return type
        if node.returns:
            return_type = self.ast_to_type(node.returns)
        else:
            return_type = Type(TypeKind.NONE)
        
        # Create IR function
        func = IRFunction(
            name=node.name,
            param_names=param_names,
            param_types=param_types,
            return_type=return_type
        )
        
        self.current_function = func
        self.temp_counter = 0
        self.label_counter = 0
        
        # Create entry block
        entry = IRBasicBlock("entry")
        self.current_block = entry
        func.add_block(entry)
        
        # Lower function body
        for stmt in node.body:
            self.visit(stmt)
        
        # Add implicit return if the current block doesn't end with a terminator
        if self.current_block and self.current_block.instructions:
            last_instr = self.current_block.instructions[-1]
            if not isinstance(last_instr, (IRReturn, IRJump, IRBranch)):
                self.emit(IRReturn(None))
        elif self.current_block:
            # Empty block, add return
            self.emit(IRReturn(None))
        
        self.module.add_function(func)
        self.current_function = None
        self.current_block = None
    
    def visit_Return(self, node: ast.Return) -> IRNode:
        """Lower return statement"""
        if node.value:
            value = self.visit(node.value)
            self.emit(IRReturn(value, node.lineno))
        else:
            self.emit(IRReturn(None, node.lineno))
    
    def visit_Assign(self, node: ast.Assign) -> IRNode:
        """Lower assignment (Week 3 Day 1: Added attribute assignment support)"""
        value = self.visit(node.value)
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Regular variable assignment: x = value
                var = IRVar(target.id, value.typ)
                self.emit(IRStore(var, value, node.lineno))
            
            elif isinstance(target, ast.Attribute):
                # Attribute assignment: obj.attr = value (Week 3 Day 1)
                obj = self.visit(target.value)
                self.emit(IRSetAttr(
                    object=obj,
                    attribute=target.attr,
                    value=value
                ))
            
            else:
                raise NotImplementedError(f"Unsupported assignment target: {type(target).__name__}")
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> IRNode:
        """Lower annotated assignment"""
        if isinstance(node.target, ast.Name):
            typ = self.ast_to_type(node.annotation)
            var = IRVar(node.target.id, typ)
            
            if node.value:
                value = self.visit(node.value)
                self.emit(IRStore(var, value, node.lineno))
    
    def visit_AugAssign(self, node: ast.AugAssign) -> IRNode:
        """Lower augmented assignment (+=, -=, etc.)"""
        if isinstance(node.target, ast.Name):
            # Load current value
            var = IRVar(node.target.id, Type(TypeKind.UNKNOWN))
            current = IRLoad(var)
            
            # Compute new value
            right = self.visit(node.value)
            op_kind = self.aug_op_to_ir(node.op)
            result = IRBinOp(op_kind, current, right, current.typ)
            
            # Store back
            self.emit(IRStore(var, result, node.lineno))
    
    def visit_If(self, node: ast.If) -> IRNode:
        """Lower if statement"""
        # Evaluate condition
        condition = self.visit(node.test)
        
        # Create labels
        then_label = self.new_label("if_then")
        else_label = self.new_label("if_else")
        end_label = self.new_label("if_end")
        
        # Branch
        if node.orelse:
            self.emit(IRBranch(condition, then_label, else_label, node.lineno))
        else:
            self.emit(IRBranch(condition, then_label, end_label, node.lineno))
        
        # Then block
        then_block = IRBasicBlock(then_label)
        self.current_function.add_block(then_block)
        self.current_block = then_block
        
        for stmt in node.body:
            self.visit(stmt)
        
        # Jump to end if no return/raise
        then_terminates = (then_block.instructions and 
                          isinstance(then_block.instructions[-1], (IRReturn, IRJump, IRBranch, IRRaise)))
        if not then_terminates:
            self.emit(IRJump(end_label))
        
        # Else block
        else_terminates = False
        if node.orelse:
            else_block = IRBasicBlock(else_label)
            self.current_function.add_block(else_block)
            self.current_block = else_block
            
            for stmt in node.orelse:
                self.visit(stmt)
            
            else_terminates = (else_block.instructions and 
                             isinstance(else_block.instructions[-1], (IRReturn, IRJump, IRBranch, IRRaise)))
            if not else_terminates:
                self.emit(IRJump(end_label))
        
        # Only create end block if needed (at least one branch doesn't terminate)
        if not (then_terminates and else_terminates):
            end_block = IRBasicBlock(end_label)
            self.current_function.add_block(end_block)
            self.current_block = end_block
    
    def visit_While(self, node: ast.While) -> IRNode:
        """Lower while loop"""
        # Create labels
        cond_label = self.new_label("while_cond")
        body_label = self.new_label("while_body")
        end_label = self.new_label("while_end")
        
        # Push loop context
        self.loop_stack.append((end_label, cond_label))
        
        # Jump to condition
        self.emit(IRJump(cond_label))
        
        # Condition block
        cond_block = IRBasicBlock(cond_label)
        self.current_function.add_block(cond_block)
        self.current_block = cond_block
        
        condition = self.visit(node.test)
        self.emit(IRBranch(condition, body_label, end_label))
        
        # Body block
        body_block = IRBasicBlock(body_label)
        self.current_function.add_block(body_block)
        self.current_block = body_block
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.emit(IRJump(cond_label))
        
        # End block
        end_block = IRBasicBlock(end_label)
        self.current_function.add_block(end_block)
        self.current_block = end_block
        
        # Pop loop context
        self.loop_stack.pop()
    
    def visit_For(self, node: ast.For) -> IRNode:
        """Lower for loop (only range() for now)"""
        # For now, only handle: for i in range(n)
        if not (isinstance(node.iter, ast.Call) and 
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == 'range'):
            raise NotImplementedError("Only range() loops supported")
        
        # Get range arguments
        if len(node.iter.args) == 1:
            start = IRConstInt(0)
            end = self.visit(node.iter.args[0])
            step = IRConstInt(1)
        elif len(node.iter.args) == 2:
            start = self.visit(node.iter.args[0])
            end = self.visit(node.iter.args[1])
            step = IRConstInt(1)
        elif len(node.iter.args) == 3:
            start = self.visit(node.iter.args[0])
            end = self.visit(node.iter.args[1])
            step = self.visit(node.iter.args[2])
        else:
            raise ValueError("Invalid range() call")
        
        # Loop variable
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")
        
        loop_var = IRVar(node.target.id, Type(TypeKind.INT))
        
        # Initialize loop variable
        self.emit(IRStore(loop_var, start))
        
        # Create labels
        cond_label = self.new_label("for_cond")
        body_label = self.new_label("for_body")
        inc_label = self.new_label("for_inc")
        end_label = self.new_label("for_end")
        
        # Push loop context
        self.loop_stack.append((end_label, inc_label))
        
        # Jump to condition
        self.emit(IRJump(cond_label))
        
        # Condition block
        cond_block = IRBasicBlock(cond_label)
        self.current_function.add_block(cond_block)
        self.current_block = cond_block
        
        i_load = IRLoad(loop_var)
        cmp = IRBinOp(IRNodeKind.LT, i_load, end, Type(TypeKind.BOOL))
        self.emit(IRBranch(cmp, body_label, end_label))
        
        # Body block
        body_block = IRBasicBlock(body_label)
        self.current_function.add_block(body_block)
        self.current_block = body_block
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.emit(IRJump(inc_label))
        
        # Increment block
        inc_block = IRBasicBlock(inc_label)
        self.current_function.add_block(inc_block)
        self.current_block = inc_block
        
        i_load2 = IRLoad(loop_var)
        new_i = IRBinOp(IRNodeKind.ADD, i_load2, step, Type(TypeKind.INT))
        self.emit(IRStore(loop_var, new_i))
        self.emit(IRJump(cond_label))
        
        # End block
        end_block = IRBasicBlock(end_label)
        self.current_function.add_block(end_block)
        self.current_block = end_block
        
        # Pop loop context
        self.loop_stack.pop()
    
    def visit_Break(self, node: ast.Break):
        """Lower break statement"""
        if not self.loop_stack:
            raise ValueError("break outside loop")
        break_label, _ = self.loop_stack[-1]
        self.emit(IRJump(break_label))
    
    def visit_Continue(self, node: ast.Continue):
        """Lower continue statement"""
        if not self.loop_stack:
            raise ValueError("continue outside loop")
        _, continue_label = self.loop_stack[-1]
        self.emit(IRJump(continue_label))
    
    def visit_Expr(self, node: ast.Expr):
        """Lower expression statement"""
        self.visit(node.value)
    
    def visit_BinOp(self, node: ast.BinOp) -> IRNode:
        """Lower binary operation"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        op_kind = self.binop_to_ir(node.op)
        
        # Determine result type with better inference
        left_type = left.typ if hasattr(left, 'typ') else Type(TypeKind.UNKNOWN)
        right_type = right.typ if hasattr(right, 'typ') else Type(TypeKind.UNKNOWN)
        
        # Type promotion rules
        if isinstance(node.op, ast.Div):
            # Division always returns float
            result_type = Type(TypeKind.FLOAT)
        elif left_type.kind == TypeKind.FLOAT or right_type.kind == TypeKind.FLOAT:
            # Any float operation returns float
            result_type = Type(TypeKind.FLOAT)
        elif left_type.kind == TypeKind.INT and right_type.kind == TypeKind.INT:
            # Int + Int = Int
            result_type = Type(TypeKind.INT)
        elif left_type.kind == TypeKind.UNKNOWN and right_type.kind != TypeKind.UNKNOWN:
            # Infer from known type
            result_type = right_type
        elif left_type.kind != TypeKind.UNKNOWN and right_type.kind == TypeKind.UNKNOWN:
            # Infer from known type
            result_type = left_type
        else:
            # Default to int
            result_type = Type(TypeKind.INT)
        
        return IRBinOp(op_kind, left, right, result_type, node.lineno)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> IRNode:
        """Lower boolean operation (and/or) with short-circuit evaluation
        
        For 'and': if left is false, return left, else return right
        For 'or': if left is true, return left, else return right
        """
        # Create blocks for short-circuit evaluation
        result_var = self.temp_var()
        result_type = Type(TypeKind.BOOL)
        
        # Evaluate first value
        result = self.visit(node.values[0])
        
        # For multiple values (a and b and c), chain them
        for i, value in enumerate(node.values[1:], 1):
            # Create blocks for short-circuit logic
            short_circuit_block = self.new_block(f"bool_short_circuit_{i}")
            continue_block = self.new_block(f"bool_continue_{i}")
            merge_block = self.new_block(f"bool_merge_{i}")
            
            if isinstance(node.op, ast.And):
                # For 'and': if result is false, short-circuit
                self.current_block.instructions.append(
                    IRBranch(result, continue_block, short_circuit_block)
                )
            else:  # ast.Or
                # For 'or': if result is true, short-circuit
                self.current_block.instructions.append(
                    IRBranch(result, short_circuit_block, continue_block)
                )
            
            # Short-circuit block: use current result
            self.current_block = short_circuit_block
            self.current_block.instructions.append(
                IRStore(result_var, result, result_type)
            )
            self.current_block.instructions.append(IRJump(merge_block))
            
            # Continue block: evaluate next value
            self.current_block = continue_block
            right = self.visit(value)
            self.current_block.instructions.append(
                IRStore(result_var, right, result_type)
            )
            self.current_block.instructions.append(IRJump(merge_block))
            
            # Merge block: load result
            self.current_block = merge_block
            result = IRLoad(result_var, result_type)
            self.current_block.instructions.append(result)
        
        return result

    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> IRNode:
        """Lower unary operation with better type inference"""
        operand = self.visit(node.operand)
        op_kind = self.unaryop_to_ir(node.op)
        
        # Determine result type based on operation
        if isinstance(node.op, ast.USub):
            # Negation preserves type
            result_type = operand.typ
        elif isinstance(node.op, ast.Not):
            # Logical not returns bool
            result_type = Type(TypeKind.BOOL)
        elif isinstance(node.op, ast.Invert):
            # Bitwise not requires and returns integer type
            if operand.typ.kind == TypeKind.INT:
                result_type = operand.typ
            else:
                result_type = Type(TypeKind.INT)
        elif isinstance(node.op, ast.UAdd):
            # Unary plus is identity - preserves type
            result_type = operand.typ
        else:
            # Unknown unary operator - preserve type
            result_type = operand.typ
        
        return IRUnaryOp(op_kind, operand, result_type, node.lineno)
    
    def visit_Compare(self, node: ast.Compare) -> IRNode:
        """Lower comparison"""
        left = self.visit(node.left)
        
        # Only handle single comparison for now
        if len(node.ops) > 1:
            raise NotImplementedError("Chained comparisons not supported")
        
        right = self.visit(node.comparators[0])
        op_kind = self.cmpop_to_ir(node.ops[0])
        
        return IRBinOp(op_kind, left, right, Type(TypeKind.BOOL), node.lineno)
    
    # ============================================================================
    # Phase 4: Advanced Language Features (Async/Await, Generators, Exceptions)
    # ============================================================================
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Lower async function definition"""
        # Extract signature
        param_nodes = []
        for arg in node.args.args:
            if arg.annotation:
                typ = self.ast_to_type(arg.annotation)
            else:
                typ = Type(TypeKind.UNKNOWN)
            param_nodes.append(IRVar(arg.arg, typ))
        
        # Return type
        if node.returns:
            return_type = self.ast_to_type(node.returns)
        else:
            return_type = Type(TypeKind.UNKNOWN)
        
        # Create IR async function
        func = IRAsyncFunction(
            name=node.name,
            params=param_nodes,
            body=[],
            return_type=return_type,
            line=node.lineno
        )
        
        self.current_function = func
        self.temp_counter = 0
        self.label_counter = 0
        
        # Create entry block
        entry = IRBasicBlock("entry")
        self.current_block = entry
        func.add_block(entry)
        
        # Lower function body
        for stmt in node.body:
            self.visit(stmt)
        
        # Add implicit return if needed
        if self.current_block and self.current_block.instructions:
            last_instr = self.current_block.instructions[-1]
            if not isinstance(last_instr, (IRReturn, IRJump, IRBranch)):
                self.emit(IRReturn(None))
        elif self.current_block:
            self.emit(IRReturn(None))
        
        self.module.add_function(func)
        self.current_function = None
        self.current_block = None
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Lower class definition - Week 2 Day 1"""
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
        
        # Create IR class
        ir_class = IRClass(
            name=node.name,
            base_classes=base_classes,
            attributes={},
            methods=[],
            line=node.lineno
        )
        
        # Process class body
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                # This is a method
                method = self._lower_method(stmt, node.name)
                if method:
                    ir_class.add_method(method)
                    # Also add method to module's function list for code generation
                    self.module.add_function(method)
            elif isinstance(stmt, ast.AnnAssign):
                # Class attribute with type annotation
                if isinstance(stmt.target, ast.Name):
                    attr_name = stmt.target.id
                    attr_type = self.ast_to_type(stmt.annotation) if stmt.annotation else Type(TypeKind.UNKNOWN)
                    ir_class.attributes[attr_name] = attr_type
            elif isinstance(stmt, ast.Assign):
                # Class attribute assignment
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        ir_class.attributes[attr_name] = Type(TypeKind.UNKNOWN)
            elif isinstance(stmt, ast.Pass):
                # Empty class body
                continue
        
        # Add class to module
        if not hasattr(self.module, 'classes'):
            self.module.classes = []
        self.module.classes.append(ir_class)
    
    def _lower_method(self, node: ast.FunctionDef, class_name: str) -> Optional[IRFunction]:
        """Lower a class method to IR function"""
        # Check if this is a generator
        if self._contains_yield(node):
            # For now, skip generator methods
            return None
        
        # Extract parameters (skip 'self' for now)
        param_names = []
        param_types = []
        for i, arg in enumerate(node.args.args):
            # First parameter is 'self' - we'll handle it specially later
            if i == 0 and arg.arg == 'self':
                continue
            
            if arg.annotation:
                typ = self.ast_to_type(arg.annotation)
            else:
                typ = Type(TypeKind.UNKNOWN)
            param_names.append(arg.arg)
            param_types.append(typ)
        
        # Return type
        if node.returns:
            return_type = self.ast_to_type(node.returns)
        else:
            return_type = Type(TypeKind.UNKNOWN)
        
        # Create IR function for method (with mangled name for now)
        method_name = f"{class_name}_{node.name}"
        func = IRFunction(
            name=method_name,
            param_names=param_names,
            param_types=param_types,
            return_type=return_type
        )
        
        # Store metadata about the method (Week 3 Day 1: Fixed for tests)
        func.metadata = {
            'original_name': node.name,
            'is_method': True,
            'class_name': class_name
        }
        # Also store as direct attributes for backward compatibility
        func.original_name = node.name
        func.is_method = True
        func.class_name = class_name
        
        # Lower method body
        old_function = self.current_function
        old_block = self.current_block
        old_temp = self.temp_counter
        old_label = self.label_counter
        
        self.current_function = func
        self.temp_counter = 0
        self.label_counter = 0
        
        # Create entry block
        entry = IRBasicBlock("entry")
        self.current_block = entry
        func.add_block(entry)
        
        # Lower method body
        for stmt in node.body:
            self.visit(stmt)
        
        # Add implicit return if needed
        if self.current_block and self.current_block.instructions:
            last_instr = self.current_block.instructions[-1]
            if not isinstance(last_instr, (IRReturn, IRJump, IRBranch, IRRaise)):
                self.emit(IRReturn(IRConstNone()))
        elif self.current_block:
            self.emit(IRReturn(IRConstNone()))
        
        # Restore state
        self.current_function = old_function
        self.current_block = old_block
        self.temp_counter = old_temp
        self.label_counter = old_label
        
        return func
    
    def visit_Await(self, node: ast.Await):
        """Lower await expression"""
        value = self.visit(node.value)
        result = self.new_temp(value.typ)
        await_node = IRAwait(value, result, node.lineno)
        self.emit(await_node)
        return result
    
    def visit_Yield(self, node: ast.Yield):
        """Lower yield expression"""
        if node.value:
            value = self.visit(node.value)
        else:
            value = IRConstInt(0)  # None placeholder
        
        yield_node = IRYield(value, node.lineno)
        self.emit(yield_node)
        return value
    
    def visit_YieldFrom(self, node: ast.YieldFrom):
        """Lower yield from expression"""
        value = self.visit(node.value)
        result = self.new_temp(value.typ)
        yield_from = IRYieldFrom(value, result, node.lineno)
        self.emit(yield_from)
        return result
    
    def visit_Try(self, node: ast.Try):
        """Lower try/except/finally statement"""
        # Create labels
        try_label = self.new_label("try_body")
        except_labels = [self.new_label(f"except_{i}") for i in range(len(node.handlers))]
        finally_label = self.new_label("finally") if node.finalbody else None
        end_label = self.new_label("try_end")
        
        # Create try block
        try_block = IRBasicBlock(try_label)
        self.current_function.add_block(try_block)
        self.current_block = try_block
        
        # Lower try body
        for stmt in node.body:
            self.visit(stmt)
        
        # Jump to finally or end (only if not already terminated)
        try_terminates = (try_block.instructions and 
                         isinstance(try_block.instructions[-1], (IRReturn, IRJump, IRBranch, IRRaise)))
        if not try_terminates:
            if finally_label:
                self.emit(IRJump(finally_label))
            else:
                self.emit(IRJump(end_label))
        
        # Create except handlers
        except_nodes = []
        for i, handler in enumerate(node.handlers):
            # Get exception type
            if handler.type:
                if isinstance(handler.type, ast.Name):
                    exc_type = handler.type.id
                else:
                    exc_type = "Exception"
            else:
                exc_type = "Exception"
            
            # Create except block
            except_block = IRBasicBlock(except_labels[i])
            self.current_function.add_block(except_block)
            self.current_block = except_block
            
            # Bind exception to variable if specified
            if handler.name:
                exc_var = IRVar(handler.name, Type(TypeKind.UNKNOWN))
                # Store caught exception (placeholder)
                self.emit(IRStore(exc_var, IRConstInt(0)))
            
            # Lower except body
            for stmt in handler.body:
                self.visit(stmt)
            
            # Jump to finally or end (only if not already terminated)
            except_terminates = (except_block.instructions and 
                               isinstance(except_block.instructions[-1], (IRReturn, IRJump, IRBranch, IRRaise)))
            if not except_terminates:
                if finally_label:
                    self.emit(IRJump(finally_label))
                else:
                    self.emit(IRJump(end_label))
            
            # Create IR except handler
            except_nodes.append(IRExcept(exc_type, except_block, i))
        
        # Create finally block
        finally_block = None
        if node.finalbody:
            finally_block = IRBasicBlock(finally_label)
            self.current_function.add_block(finally_block)
            self.current_block = finally_block
            
            for stmt in node.finalbody:
                self.visit(stmt)
            
            # Jump to end (only if not already terminated)
            finally_terminates = (finally_block.instructions and 
                                isinstance(finally_block.instructions[-1], (IRReturn, IRJump, IRBranch, IRRaise)))
            if not finally_terminates:
                self.emit(IRJump(end_label))
        
        # Note: IRTry node emission removed - exception handling is done
        # through control flow blocks, not as instruction nodes
        
        # Create end block
        end_block = IRBasicBlock(end_label)
        self.current_function.add_block(end_block)
        self.current_block = end_block
    
    def visit_Raise(self, node: ast.Raise):
        """Lower raise statement"""
        if node.exc:
            exc = self.visit(node.exc)
        else:
            exc = None  # Re-raise
        
        raise_node = IRRaise(exc, node.lineno)
        self.emit(raise_node)
    
    def visit_With(self, node: ast.With):
        """Lower with statement (context manager)"""
        # For now, handle single context manager
        if len(node.items) != 1:
            raise NotImplementedError("Multiple context managers not yet supported")
        
        item = node.items[0]
        context_expr = self.visit(item.context_expr)
        
        # Optional variable binding (store context result)
        if item.optional_vars:
            if isinstance(item.optional_vars, ast.Name):
                var_name = item.optional_vars.id
                # Create variable and store context expression
                var = IRVar(var_name, context_expr.typ if hasattr(context_expr, 'typ') else Type(TypeKind.UNKNOWN))
                self.emit(IRStore(var, context_expr))
            else:
                raise NotImplementedError("Complex context manager targets not supported")
        
        # Lower with body directly (no need for separate blocks in simple case)
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_Name(self, node: ast.Name) -> IRNode:
        """Lower variable reference"""
        # Look up type from symbol table
        symbol = self.symbol_table.lookup(node.id)
        if symbol:
            typ = symbol.typ
        else:
            typ = Type(TypeKind.UNKNOWN)
        
        var = IRVar(node.id, typ)
        return IRLoad(var, node.lineno)
    
    def visit_Constant(self, node: ast.Constant) -> IRNode:
        """Lower constant with support for more types"""
        if isinstance(node.value, bool):
            # Check bool before int (bool is subclass of int in Python)
            return IRConstBool(node.value, node.lineno)
        elif isinstance(node.value, int):
            return IRConstInt(node.value, node.lineno)
        elif isinstance(node.value, float):
            return IRConstFloat(node.value, node.lineno)
        elif isinstance(node.value, str):
            # Store string as constant - backend will handle global string allocation
            return IRConstStr(node.value, node.lineno)
        elif node.value is None:
            # None constant - backend should map to null/nullptr
            return IRConstNone(node.lineno)
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(node.value).__name__}")

    
    def visit_Call(self, node: ast.Call) -> IRNode:
        """Lower function call (Week 3 Day 1: Added class instantiation support)"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check if this is a class instantiation (Week 3 Day 1)
            # Look for class in symbol table
            try:
                symbol = self.symbol_table.lookup(func_name)
                if hasattr(symbol, 'kind') and symbol.kind == 'class':
                    # This is class instantiation: Point(10, 20)
                    args = [self.visit(arg) for arg in node.args]
                    result_var = IRVar(
                        self.new_temp(),
                        Type(TypeKind.UNKNOWN)  # TODO: Use class type
                    )
                    
                    return IRNewObject(
                        class_name=func_name,
                        args=args,
                        result=result_var
                    )
            except (AttributeError, KeyError):
                # Not in symbol table or not a class, treat as function
                pass
            
            # Direct function call: func(args)
            args = [self.visit(arg) for arg in node.args]
            
            # Look up function signature for return type
            # For now, use unknown type
            return_type = Type(TypeKind.UNKNOWN)
            
            return IRCall(func_name, args, return_type, node.lineno)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method(args)
            obj = self.visit(node.func.value)
            method_name = node.func.attr
            args = [self.visit(arg) for arg in node.args]
            
            # For now, treat method calls as function calls with obj as first arg
            # Format: obj_method(obj, args...)
            func_name = f"{method_name}"  # Simplified - just use method name
            all_args = [obj] + args
            return_type = Type(TypeKind.UNKNOWN)
            
            return IRCall(func_name, all_args, return_type, node.lineno)
        else:
            raise NotImplementedError(f"Unsupported call type: {type(node.func).__name__}")
    
    def visit_Attribute(self, node: ast.Attribute) -> IRNode:
        """
        Lower attribute access (Week 3 Day 1)
        
        Handles:
        - obj.attr (Load context) → IRGetAttr
        - obj.attr (Store context) → handled in visit_Assign
        """
        obj = self.visit(node.value)
        
        # In Load context, generate IRGetAttr
        if isinstance(node.ctx, ast.Load):
            # Create a temporary variable for the result
            result_var = self.new_temp(Type(TypeKind.UNKNOWN))
            
            return IRGetAttr(
                object=obj,
                attribute=node.attr,
                result=result_var
            )
        
        # Store context is handled by visit_Assign
        # Just return the object for now
        return obj
    
    # Helper methods
    
    def ast_to_type(self, node: Optional[ast.expr]) -> Type:
        """Convert AST type annotation to Type"""
        if node is None:
            return Type(TypeKind.UNKNOWN)
        
        if isinstance(node, ast.Name):
            type_map = {
                'int': TypeKind.INT,
                'float': TypeKind.FLOAT,
                'bool': TypeKind.BOOL,
                'str': TypeKind.STR,
            }
            return Type(type_map.get(node.id, TypeKind.UNKNOWN))
        
        return Type(TypeKind.UNKNOWN)
    
    def binop_to_ir(self, op: ast.operator) -> IRNodeKind:
        """Convert AST binop to IR kind"""
        mapping = {
            ast.Add: IRNodeKind.ADD,
            ast.Sub: IRNodeKind.SUB,
            ast.Mult: IRNodeKind.MUL,
            ast.Div: IRNodeKind.DIV,
            ast.FloorDiv: IRNodeKind.FLOORDIV,
            ast.Mod: IRNodeKind.MOD,
            ast.Pow: IRNodeKind.POW,
        }
        return mapping.get(type(op), IRNodeKind.ADD)
    
    def unaryop_to_ir(self, op: ast.unaryop) -> IRNodeKind:
        """Convert AST unaryop to IR kind"""
        mapping = {
            ast.USub: IRNodeKind.NEG,
            ast.Not: IRNodeKind.NOT,
            ast.Invert: IRNodeKind.INVERT,
        }
        return mapping.get(type(op), IRNodeKind.NEG)
    
    def cmpop_to_ir(self, op: ast.cmpop) -> IRNodeKind:
        """Convert AST cmpop to IR kind"""
        mapping = {
            ast.Eq: IRNodeKind.EQ,
            ast.NotEq: IRNodeKind.NE,
            ast.Lt: IRNodeKind.LT,
            ast.LtE: IRNodeKind.LE,
            ast.Gt: IRNodeKind.GT,
            ast.GtE: IRNodeKind.GE,
        }
        return mapping.get(type(op), IRNodeKind.EQ)
    
    def aug_op_to_ir(self, op: ast.operator) -> IRNodeKind:
        """Convert augmented assignment op to IR kind"""
        return self.binop_to_ir(op)


def lower_to_ir(tree: ast.Module, symbol_table: SymbolTable) -> IRModule:
    """
    Convert Python AST to typed IR
    
    Args:
        tree: Python AST module
        symbol_table: Symbol table with type information
        
    Returns:
        IR module
    """
    lowering = IRLowering(symbol_table)
    return lowering.visit_Module(tree)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("AST TO IR LOWERING - Phase 1.2")
    print("=" * 80)
    
    # Test 1: Simple function
    print("\n--- Test 1: Simple addition function ---")
    source1 = """
def add(x: int, y: int) -> int:
    result: int = x + y
    return result
"""
    
    tree1 = ast.parse(source1)
    
    # Build symbol table (simplified)
    from compiler.frontend.symbols import SymbolTable
    symbol_table1 = SymbolTable(name="global")
    
    lowering1 = IRLowering(symbol_table1)
    module1 = lowering1.visit_Module(tree1)
    
    print(module1)
    
    # Test 2: Function with control flow
    print("\n" + "=" * 80)
    print("--- Test 2: Max function with if ---")
    print("=" * 80)
    source2 = """
def max_value(a: int, b: int) -> int:
    if a > b:
        return a
    else:
        return b
"""
    
    tree2 = ast.parse(source2)
    symbol_table2 = SymbolTable(name="global")
    
    lowering2 = IRLowering(symbol_table2)
    module2 = lowering2.visit_Module(tree2)
    
    print(module2)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("--- Test 3: Sum with for loop ---")
    print("=" * 80)
    source3 = """
def sum_range(n: int) -> int:
    total: int = 0
    for i in range(n):
        total += i
    return total
"""
    
    tree3 = ast.parse(source3)
    symbol_table3 = SymbolTable(name="global")
    
    lowering3 = IRLowering(symbol_table3)
    module3 = lowering3.visit_Module(tree3)
    
    print(module3)
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.2 AST to IR Lowering Complete!")
    print("=" * 80)
    print("\n🎉 Phase 1.2 (IR) COMPLETE!")
    print("Next: Phase 1.3 - LLVM Backend")
