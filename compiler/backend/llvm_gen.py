"""
LLVM IR Generation - Convert typed IR to LLVM IR

This module generates LLVM IR from our custom typed IR using llvmlite.
It handles:
- Type mapping
- Instruction generation
- Control flow
- Function generation

Phase: 1.3 (Backend)
"""

from llvmlite import ir
from typing import Dict, Optional
import llvmlite.binding as llvm

from compiler.ir.ir_nodes import *
from compiler.frontend.semantic import Type, TypeKind


class LLVMCodeGen:
    """
    Generates LLVM IR from typed IR
    
    Uses llvmlite to create LLVM IR suitable for optimization
    and native code generation.
    """
    
    def __init__(self):
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Create LLVM module
        self.module = ir.Module(name="main_module")
        self.module.triple = llvm.get_default_triple()
        
        # Builder for generating instructions
        self.builder: Optional[ir.IRBuilder] = None
        
        # Current function being generated
        self.current_function: Optional[ir.Function] = None
        
        # Variable mapping: name -> llvm value
        self.variables: Dict[str, ir.Value] = {}
        
        # Block mapping: label -> llvm block
        self.blocks: Dict[str, ir.Block] = {}
        
        # Type cache
        self.type_cache: Dict[str, ir.Type] = {}
        
        # Class struct types (Week 2 Day 2)
        self.class_types: Dict[str, ir.Type] = {}
        
        # Runtime functions (Week 2 Day 3)
        self.malloc_func = None
        self.free_func = None
    
    def declare_runtime_functions(self):
        """Declare runtime functions like malloc and free"""
        # malloc: i8* malloc(i64)
        if self.malloc_func is None:
            malloc_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(64)])
            self.malloc_func = ir.Function(self.module, malloc_type, name="malloc")
        
        # free: void free(i8*)
        if self.free_func is None:
            free_type = ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()])
            self.free_func = ir.Function(self.module, free_type, name="free")
    
    def generate_class_struct(self, cls):
        """
        Generate LLVM struct type for a class - Week 2 Day 2
        Week 3 Day 2: Fixed context isolation
        """
        struct_name = f"class.{cls.name}"
        
        # Check if struct already exists in this module
        if cls.name in self.class_types:
            return  # Already generated
        
        # Collect attribute types
        attr_types = []
        attr_names = []
        
        # Add attributes
        for attr_name, attr_type in cls.attributes.items():
            llvm_type = self.type_to_llvm(attr_type)
            attr_types.append(llvm_type)
            attr_names.append(attr_name)
        
        # If no attributes, add a dummy i32 field to make struct valid
        if not attr_types:
            attr_types.append(ir.IntType(32))
            attr_names.append("_dummy")
        
        # Create the struct type
        try:
            struct_type = self.module.context.get_identified_type(struct_name)
            if struct_type.is_opaque:
                # Only set body if not already set
                struct_type.set_body(*attr_types)
        except KeyError:
            # Type doesn't exist, create it
            struct_type = self.module.context.get_identified_type(struct_name)
            struct_type.set_body(*attr_types)
        
        # Store in class types map
        self.class_types[cls.name] = struct_type
        
        # Store attribute names for later use
        if not hasattr(self, 'class_attr_names'):
            self.class_attr_names = {}
        self.class_attr_names[cls.name] = attr_names
        
        return struct_type
    
    def type_to_llvm(self, typ: Type) -> ir.Type:
        """Convert our Type to LLVM type"""
        type_str = str(typ)
        
        if type_str in self.type_cache:
            return self.type_cache[type_str]
        
        llvm_type = None
        
        if typ.kind == TypeKind.INT:
            llvm_type = ir.IntType(64)  # 64-bit int
        elif typ.kind == TypeKind.FLOAT:
            llvm_type = ir.DoubleType()
        elif typ.kind == TypeKind.BOOL:
            llvm_type = ir.IntType(1)  # i1
        elif typ.kind == TypeKind.NONE:
            llvm_type = ir.VoidType()
        else:
            # Default to i64 for unknown types
            llvm_type = ir.IntType(64)
        
        self.type_cache[type_str] = llvm_type
        return llvm_type
    
    def generate_module(self, ir_module: IRModule) -> str:
        """
        Generate LLVM IR for entire module
        
        Args:
            ir_module: Our typed IR module
            
        Returns:
            LLVM IR as string
        """
        # Declare runtime functions (Week 2 Day 3)
        self.declare_runtime_functions()
        
        # Generate class struct types first (Week 2 Day 2)
        if hasattr(ir_module, 'classes'):
            for cls in ir_module.classes:
                self.generate_class_struct(cls)
        
        # Generate functions
        for func in ir_module.functions:
            # Check if this is an async/generator function
            if isinstance(func, IRAsyncFunction):
                self.generate_async_function(func)
            else:
                self.generate_function(func)
        
        # Return LLVM IR string
        return str(self.module)
    
    def generate_function(self, func: IRFunction):
        """Generate LLVM IR for function"""
        # Convert parameter types
        param_types = [self.type_to_llvm(t) for t in func.param_types]
        
        # Convert return type
        return_type = self.type_to_llvm(func.return_type)
        
        # Create function type
        func_type = ir.FunctionType(return_type, param_types)
        
        # Create function
        llvm_func = ir.Function(self.module, func_type, name=func.name)
        
        # Set parameter names
        for i, param_name in enumerate(func.param_names):
            llvm_func.args[i].name = param_name
        
        self.current_function = llvm_func
        self.variables = {}
        self.blocks = {}
        
        # Create entry block
        entry_block = llvm_func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        
        # Allocate space for parameters and local variables
        for i, (param_name, param_type) in enumerate(zip(func.param_names, func.param_types)):
            llvm_type = self.type_to_llvm(param_type)
            alloca = self.builder.alloca(llvm_type, name=param_name)
            self.builder.store(llvm_func.args[i], alloca)
            self.variables[param_name] = alloca
        
        # Pre-create all blocks
        for block in func.blocks[1:]:  # Skip entry, already created
            llvm_block = llvm_func.append_basic_block(name=block.name)
            self.blocks[block.name] = llvm_block
        
        self.blocks["entry"] = entry_block
        
        # Generate instructions for each block
        # Track last computed value for each block
        self.last_value: Optional[ir.Value] = None
        
        for block in func.blocks:
            if block.name != "entry":
                self.builder = ir.IRBuilder(self.blocks[block.name])
            else:
                self.builder = ir.IRBuilder(entry_block)
            
            for instr in block.instructions:
                result = self.generate_instruction(instr)
                if result is not None:
                    self.last_value = result
    
    def generate_instruction(self, instr: IRNode) -> Optional[ir.Value]:
        """Generate LLVM instruction from IR node"""
        if isinstance(instr, IRConstInt):
            llvm_type = self.type_to_llvm(instr.typ)
            return ir.Constant(llvm_type, instr.value)
        
        elif isinstance(instr, IRConstFloat):
            llvm_type = self.type_to_llvm(instr.typ)
            return ir.Constant(llvm_type, instr.value)
        
        elif isinstance(instr, IRConstBool):
            llvm_type = self.type_to_llvm(instr.typ)
            return ir.Constant(llvm_type, int(instr.value))
        
        elif isinstance(instr, IRConstStr):
            # For strings, represent as i64 pointer for now (simplified)
            # In full implementation: would create global string constant
            return ir.Constant(ir.IntType(64), 0)  # Placeholder
        
        elif isinstance(instr, IRVar):
            # IRVar represents a variable reference - load it
            var_name = instr.name
            if var_name in self.variables:
                return self.builder.load(self.variables[var_name], name=f"{var_name}_load")
            else:
                # Variable not found, allocate and return a placeholder
                llvm_type = self.type_to_llvm(instr.typ)
                alloca = self.builder.alloca(llvm_type, name=var_name)
                self.variables[var_name] = alloca
                return self.builder.load(alloca, name=f"{var_name}_load")
        
        elif isinstance(instr, IRLoad):
            var_name = instr.var.name
            if var_name in self.variables:
                return self.builder.load(self.variables[var_name], name=f"{var_name}_load")
            else:
                # Variable not found, allocate it
                llvm_type = self.type_to_llvm(instr.var.typ)
                alloca = self.builder.alloca(llvm_type, name=var_name)
                self.variables[var_name] = alloca
                return self.builder.load(alloca, name=f"{var_name}_load")
        
        elif isinstance(instr, IRStore):
            var_name = instr.var.name
            value = self.generate_instruction(instr.value)
            
            # Allocate if doesn't exist
            if var_name not in self.variables:
                llvm_type = self.type_to_llvm(instr.var.typ)
                alloca = self.builder.alloca(llvm_type, name=var_name)
                self.variables[var_name] = alloca
            
            # Get target type and convert value if needed
            target_ptr = self.variables[var_name]
            target_type = target_ptr.type.pointee
            
            # Type conversion if needed
            if value.type != target_type:
                if isinstance(value.type, ir.DoubleType) and isinstance(target_type, ir.IntType):
                    # Float to int: truncate
                    value = self.builder.fptosi(value, target_type, name="float_to_int")
                elif isinstance(value.type, ir.IntType) and isinstance(target_type, ir.DoubleType):
                    # Int to float: convert
                    value = self.builder.sitofp(value, target_type, name="int_to_float")
            
            self.builder.store(value, target_ptr)
            return None
        
        elif isinstance(instr, IRBinOp):
            left = self.generate_instruction(instr.left)
            right = self.generate_instruction(instr.right)
            
            # Determine if we're working with integers or floats based on result type
            is_float = instr.typ.kind == TypeKind.FLOAT
            is_comparison = instr.kind in (IRNodeKind.EQ, IRNodeKind.NE, IRNodeKind.LT, 
                                          IRNodeKind.LE, IRNodeKind.GT, IRNodeKind.GE)
            
            # Type promotion: if result should be float but operands are int, convert them
            if is_float and isinstance(left.type, ir.IntType):
                left = self.builder.sitofp(left, ir.DoubleType(), name="left_to_float")
            if is_float and isinstance(right.type, ir.IntType):
                right = self.builder.sitofp(right, ir.DoubleType(), name="right_to_float")
            
            # Comparisons
            if is_comparison:
                # Check actual operand types (after potential conversion)
                if isinstance(left.type, ir.DoubleType):
                    op_map = {
                        IRNodeKind.EQ: "==",
                        IRNodeKind.NE: "!=",
                        IRNodeKind.LT: "<",
                        IRNodeKind.LE: "<=",
                        IRNodeKind.GT: ">",
                        IRNodeKind.GE: ">="
                    }
                    return self.builder.fcmp_ordered(op_map[instr.kind], left, right, name=instr.kind.value)
                else:
                    op_map = {
                        IRNodeKind.EQ: "==",
                        IRNodeKind.NE: "!=",
                        IRNodeKind.LT: "<",
                        IRNodeKind.LE: "<=",
                        IRNodeKind.GT: ">",
                        IRNodeKind.GE: ">="
                    }
                    return self.builder.icmp_signed(op_map[instr.kind], left, right, name=instr.kind.value)
            
            # Arithmetic operations
            # After type promotion, check actual operand types
            operand_is_float = isinstance(left.type, ir.DoubleType)
            
            if operand_is_float:
                op_map = {
                    IRNodeKind.ADD: self.builder.fadd,
                    IRNodeKind.SUB: self.builder.fsub,
                    IRNodeKind.MUL: self.builder.fmul,
                    IRNodeKind.DIV: self.builder.fdiv,
                    IRNodeKind.MOD: self.builder.frem,
                }
                if instr.kind in op_map:
                    return op_map[instr.kind](left, right, name=instr.kind.value)
            else:
                # Integer operations
                if instr.kind == IRNodeKind.ADD:
                    return self.builder.add(left, right, name="add")
                elif instr.kind == IRNodeKind.SUB:
                    return self.builder.sub(left, right, name="sub")
                elif instr.kind == IRNodeKind.MUL:
                    return self.builder.mul(left, right, name="mul")
                elif instr.kind == IRNodeKind.DIV:
                    # This shouldn't happen anymore since we convert above
                    # But keep as fallback
                    left_f = self.builder.sitofp(left, ir.DoubleType())
                    right_f = self.builder.sitofp(right, ir.DoubleType())
                    return self.builder.fdiv(left_f, right_f, name="div")
                elif instr.kind == IRNodeKind.FLOORDIV:
                    return self.builder.sdiv(left, right, name="floordiv")
                elif instr.kind == IRNodeKind.MOD:
                    return self.builder.srem(left, right, name="mod")
                elif instr.kind == IRNodeKind.POW:
                    # Power - call intrinsic or implement manually
                    # For now, not supported
                    pass
            
            # Fallback for unsupported operations
            print(f"Warning: Unsupported operation {instr.kind} for type {instr.typ}")
            return self.builder.add(left, right, name="unsupported_op")
        
        elif isinstance(instr, IRUnaryOp):
            operand = self.generate_instruction(instr.operand)
            
            if instr.kind == IRNodeKind.NEG:
                # Check the LLVM type of the operand
                if isinstance(operand.type, (ir.IntType, ir.PointerType)):
                    return self.builder.neg(operand, name="neg")
                else:  # Float type
                    return self.builder.fneg(operand, name="fneg")
            elif instr.kind == IRNodeKind.NOT:
                # Logical not - always returns i1 (bool)
                # If operand is not i1, convert it first
                if not isinstance(operand.type, ir.IntType) or operand.type.width != 1:
                    # Convert to bool by comparing with zero
                    zero = ir.Constant(operand.type, 0)
                    operand = self.builder.icmp_signed('!=', operand, zero, name="tobool")
                return self.builder.not_(operand, name="not")
            elif instr.kind == IRNodeKind.INVERT:
                # Bitwise not - XOR with -1
                neg_one = ir.Constant(operand.type, -1)
                return self.builder.xor(operand, neg_one, name="invert")
        
        elif isinstance(instr, IRReturn):
            # Only emit if block not already terminated
            if not self.builder.block.is_terminated:
                if instr.value:
                    value = self.generate_instruction(instr.value)
                    self.builder.ret(value)
                else:
                    self.builder.ret_void()
            return None
        
        elif isinstance(instr, IRJump):
            target_block = self.blocks[instr.target]
            self.builder.branch(target_block)
            return None
        
        elif isinstance(instr, IRBranch):
            condition = self.generate_instruction(instr.condition)
            true_block = self.blocks[instr.true_label]
            false_block = self.blocks[instr.false_label]
            self.builder.cbranch(condition, true_block, false_block)
            return None
        
        elif isinstance(instr, IRCall):
            # Look up function - use try/except to handle missing functions
            try:
                callee = self.module.get_global(instr.func_name)
            except KeyError:
                # Function not found - create external declaration for testing
                # In production: would be an error or require explicit declaration
                arg_types = [self.type_to_llvm(arg.typ) if hasattr(arg, 'typ') else ir.IntType(64) 
                            for arg in instr.args]
                return_type = self.type_to_llvm(instr.typ) if hasattr(instr, 'typ') else ir.IntType(64)
                func_type = ir.FunctionType(return_type, arg_types)
                callee = ir.Function(self.module, func_type, name=instr.func_name)
            
            args = [self.generate_instruction(arg) for arg in instr.args]
            return self.builder.call(callee, args, name="call")

        
        # Phase 4: Async/Await Support
        elif isinstance(instr, IRAsyncFunction):
            return self.generate_async_function(instr)
        
        elif isinstance(instr, IRAwait):
            return self.generate_await(instr)
        
        # Phase 4: Generator Support
        elif isinstance(instr, IRYield):
            return self.generate_yield(instr)
        
        elif isinstance(instr, IRYieldFrom):
            return self.generate_yield_from(instr)
        
        # Phase 4: Exception Handling
        elif isinstance(instr, IRTry):
            return self.generate_try(instr)
        
        elif isinstance(instr, IRRaise):
            return self.generate_raise(instr)
        
        # Phase 4: Context Managers
        elif isinstance(instr, IRWith):
            return self.generate_with(instr)
        
        # Week 2: OOP Support (Days 3-5)
        elif isinstance(instr, IRNewObject):
            return self.generate_new_object(instr)
        
        elif isinstance(instr, IRGetAttr):
            return self.generate_get_attr(instr)
        
        elif isinstance(instr, IRSetAttr):
            return self.generate_set_attr(instr)
        
        elif isinstance(instr, IRMethodCall):
            return self.generate_method_call(instr)
        
        return None
    
    def optimize(self, optimization_level: int = 2):
        """
        Optimize LLVM IR
        
        Args:
            optimization_level: 0-3 (0=none, 3=aggressive)
        """
        # Parse IR to module
        llvm_ir = str(self.module)
        llvm_module = llvm.parse_assembly(llvm_ir)
        
        # Create pass manager
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = optimization_level
        
        pm = llvm.create_module_pass_manager()
    
    # ========== Phase 4: Advanced Feature Code Generation ==========
    
    def generate_async_function(self, async_func: 'IRAsyncFunction') -> ir.Value:
        """
        Generate LLVM coroutine for async function using coroutine intrinsics.
        
        Async functions are transformed into LLVM coroutines that can suspend
        and resume execution. Uses @llvm.coro.* intrinsics.
        
        Strategy:
        1. Create coroutine frame with @llvm.coro.id and @llvm.coro.begin
        2. Transform function body to support suspend points
        3. Add @llvm.coro.suspend at await points
        4. Add @llvm.coro.end to finalize
        """
        # Convert parameter types
        param_types = [self.type_to_llvm(param.typ) for param in async_func.params]
        
        # Convert return type
        return_type = self.type_to_llvm(async_func.typ)
        
        # Create function type
        func_type = ir.FunctionType(return_type, param_types)
        
        # Create function
        llvm_func = ir.Function(self.module, func_type, name=async_func.name)
        
        # Set parameter names
        for i, param in enumerate(async_func.params):
            llvm_func.args[i].name = param.name
        
        self.current_function = llvm_func
        self.variables = {}
        self.blocks = {}
        
        # Create entry block
        entry_block = llvm_func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        
        # Declare LLVM coroutine intrinsics (only if not already declared)
        coro_id_name = "llvm.coro.id"
        if coro_id_name not in self.module.globals:
            coro_id_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(8).as_pointer(),
                                                             ir.IntType(8).as_pointer(), ir.IntType(8).as_pointer()])
            coro_id = ir.Function(self.module, coro_id_type, name=coro_id_name)
        else:
            coro_id = self.module.get_global(coro_id_name)
        
        coro_begin_name = "llvm.coro.begin"
        if coro_begin_name not in self.module.globals:
            coro_begin_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(32), ir.IntType(8).as_pointer()])
            coro_begin = ir.Function(self.module, coro_begin_type, name=coro_begin_name)
        else:
            coro_begin = self.module.get_global(coro_begin_name)
        
        coro_size_name = "llvm.coro.size.i64"
        if coro_size_name not in self.module.globals:
            coro_size_type = ir.FunctionType(ir.IntType(64), [])
            coro_size = ir.Function(self.module, coro_size_type, name=coro_size_name)
        else:
            coro_size = self.module.get_global(coro_size_name)
        
        # Allocate coroutine frame
        size = self.builder.call(coro_size, [], name="coro.size")
        frame = self.builder.alloca(ir.IntType(8), size, name="coro.frame")
        
        # Initialize coroutine
        null = ir.Constant(ir.IntType(8).as_pointer(), None)
        align = ir.Constant(ir.IntType(32), 0)
        id_val = self.builder.call(coro_id, [align, null, null, null], name="coro.id")
        handle = self.builder.call(coro_begin, [id_val, frame], name="coro.hdl")
        
        # Store coroutine handle for later await operations
        self.variables[f"_coro_handle_{async_func.name}"] = handle
        
        # Generate function body (blocks)
        for block in async_func.body:
            if isinstance(block, IRBasicBlock):
                llvm_block = llvm_func.append_basic_block(name=block.name)
                self.blocks[block.name] = llvm_block
        
        # Generate instructions for each block
        for block in async_func.body:
            if isinstance(block, IRBasicBlock):
                self.builder = ir.IRBuilder(self.blocks[block.name])
                for instr in block.instructions:
                    self.generate_instruction(instr)
        
        return handle
    
    def generate_await(self, await_node: 'IRAwait') -> ir.Value:
        """
        Generate LLVM code for await expression using @llvm.coro.suspend.
        
        An await suspends the coroutine and yields control back to the caller.
        When resumed, execution continues after the suspend point.
        """
        # Declare suspend intrinsic
        suspend_type = ir.FunctionType(ir.IntType(8), [ir.IntType(32), ir.IntType(1)])
        coro_suspend = ir.Function(self.module, suspend_type, name="llvm.coro.suspend")
        
        # Generate coroutine value to await
        coro_value = self.generate_instruction(await_node.coroutine)
        
        # Suspend coroutine
        save_value = ir.Constant(ir.IntType(32), 0)
        final = ir.Constant(ir.IntType(1), 0)
        suspend_result = self.builder.call(coro_suspend, [save_value, final], name="await.suspend")
        
        # Check suspend result: 0 = suspended, -1 = done, 1 = error
        zero = ir.Constant(ir.IntType(8), 0)
        is_suspended = self.builder.icmp_signed("==", suspend_result, zero, name="is.suspended")
        
        # Create resume block (executed when coroutine resumes)
        resume_block = self.current_function.append_basic_block(name="await.resume")
        cleanup_block = self.current_function.append_basic_block(name="await.cleanup")
        
        self.builder.cbranch(is_suspended, resume_block, cleanup_block)
        
        # Resume block: extract result from coroutine
        self.builder = ir.IRBuilder(resume_block)
        # In full implementation: extract result from coroutine frame
        result_type = self.type_to_llvm(await_node.result_type)
        result = ir.Constant(result_type, 0)  # Placeholder
        
        return result
    
    def generate_yield(self, yield_node: 'IRYield') -> ir.Value:
        """
        Generate LLVM code for yield expression.
        
        Yield is implemented as a state machine transformation:
        1. Save local state to coroutine frame
        2. Return yielded value
        3. On resume, restore state and continue
        """
        # Generate value to yield
        value = self.generate_instruction(yield_node.value) if yield_node.value else None
        
        # In a generator, we transform the function into a state machine
        # Each yield point gets a unique state ID
        if not hasattr(self, '_yield_state_counter'):
            self._yield_state_counter = 0
        
        state_id = self._yield_state_counter
        self._yield_state_counter += 1
        
        # Save current state ID to generator frame
        # (In full implementation: store all local variables to frame)
        state_ptr = self.variables.get("_gen_state")
        if state_ptr:
            state_const = ir.Constant(ir.IntType(32), state_id)
            self.builder.store(state_const, state_ptr)
        
        # Return yielded value (or None for yield without value)
        if value:
            return value
        else:
            return ir.Constant(ir.IntType(32), 0)  # Placeholder for None
    
    def generate_yield_from(self, yield_from: 'IRYieldFrom') -> ir.Value:
        """
        Generate LLVM code for 'yield from' delegation.
        
        Yield from delegates to a sub-iterator, yielding all its values.
        Implemented as a loop that calls the sub-iterator.
        """
        # Generate sub-iterator
        iterator = self.generate_instruction(yield_from.iterator)
        
        # Create loop to iterate and yield all values
        loop_header = self.current_function.append_basic_block(name="yield_from.loop")
        loop_body = self.current_function.append_basic_block(name="yield_from.body")
        loop_exit = self.current_function.append_basic_block(name="yield_from.exit")
        
        self.builder.branch(loop_header)
        
        # Loop header: check if iterator has more items
        self.builder = ir.IRBuilder(loop_header)
        # In full implementation: call iterator's __next__ method
        has_next = ir.Constant(ir.IntType(1), 1)  # Placeholder
        self.builder.cbranch(has_next, loop_body, loop_exit)
        
        # Loop body: yield next value
        self.builder = ir.IRBuilder(loop_body)
        # next_value = iterator.__next__()
        # yield next_value
        self.builder.branch(loop_header)
        
        # Loop exit
        self.builder = ir.IRBuilder(loop_exit)
        return ir.Constant(ir.IntType(32), 0)  # Placeholder
    
    def generate_try(self, try_node: 'IRTry') -> ir.Value:
        """
        Generate LLVM exception handling using invoke/landingpad.
        
        Strategy:
        1. Use 'invoke' instead of 'call' for operations that might throw
        2. Add 'landingpad' instruction to catch exceptions
        3. Route to appropriate except handler based on exception type
        4. Ensure finally block always executes
        """
        # Create basic blocks
        try_block = self.current_function.append_basic_block(name="try")
        landing_pad = self.current_function.append_basic_block(name="lpad")
        finally_block = self.current_function.append_basic_block(name="finally") if try_node.finally_block else None
        continue_block = self.current_function.append_basic_block(name="try.cont")
        
        # Generate try block body
        self.builder.branch(try_block)
        self.builder = ir.IRBuilder(try_block)
        
        for stmt in try_node.body:
            self.generate_instruction(stmt)
        
        # If no exception, jump to finally (if exists) or continue
        if finally_block:
            self.builder.branch(finally_block)
        else:
            self.builder.branch(continue_block)
        
        # Landing pad: exception caught here
        self.builder = ir.IRBuilder(landing_pad)
        
        # Create landingpad instruction
        # landingpad { i8*, i32 }
        exception_type = ir.LiteralStructType([ir.IntType(8).as_pointer(), ir.IntType(32)])
        
        # In llvmlite, landingpad takes just the type, not personality
        lpad_instr = self.builder.landingpad(exception_type, name="lpad.val")
        
        # Set personality function for the whole function
        if not hasattr(self.current_function, '_personality_set'):
            personality_type = ir.FunctionType(ir.IntType(32), [], var_arg=True)
            try:
                personality = self.module.get_global("__gxx_personality_v0")
            except KeyError:
                personality = ir.Function(self.module, personality_type, name="__gxx_personality_v0")
            self.current_function._personality_set = True
        
        # Make this a cleanup landingpad (catches all exceptions)
        # This is simpler than trying to use type-specific catch clauses
        lpad_instr.cleanup = True
        
        # Extract exception info
        exc_ptr = self.builder.extract_value(lpad_instr, 0, name="exc.ptr")
        exc_sel = self.builder.extract_value(lpad_instr, 1, name="exc.sel")
        
        # Route to appropriate except handler
        for i, except_node in enumerate(try_node.except_blocks):
            except_bb = self.current_function.append_basic_block(name=f"except.{i}")
            self.blocks[f"except_{i}"] = except_bb
            
            # Generate except handler body
            self.builder = ir.IRBuilder(except_bb)
            
            # Store exception in variable if named
            if except_node.var_name:
                self.variables[except_node.var_name] = exc_ptr
            
            for stmt in except_node.handler:
                self.generate_instruction(stmt)
            
            # After handler, jump to finally or continue
            if finally_block:
                self.builder.branch(finally_block)
            else:
                self.builder.branch(continue_block)
        
        # Generate finally block if present
        if finally_block and try_node.finally_block:
            self.builder = ir.IRBuilder(finally_block)
            for stmt in try_node.finally_block.body:
                self.generate_instruction(stmt)
            self.builder.branch(continue_block)
        
        # Continue execution
        self.builder = ir.IRBuilder(continue_block)
        return None
    
    def generate_raise(self, raise_node: 'IRRaise') -> ir.Value:
        """
        Generate LLVM code to raise an exception.
        
        Uses __cxa_throw (C++ ABI) to throw exception.
        """
        # Declare exception throwing function
        throw_type = ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer(),
                                                      ir.IntType(8).as_pointer(),
                                                      ir.IntType(8).as_pointer()])
        cxa_throw = ir.Function(self.module, throw_type, name="__cxa_throw")
        
        # Generate exception object
        if raise_node.exception:
            exc_value = self.generate_instruction(raise_node.exception)
            if not exc_value:  # If generation failed, use placeholder
                exc_value = ir.Constant(ir.IntType(8).as_pointer(), None)
            elif not isinstance(exc_value.type, ir.PointerType):
                # Convert to i8* if not already a pointer
                exc_value = self.builder.inttoptr(exc_value, ir.IntType(8).as_pointer(), name="exc_ptr")
        else:
            # Re-raise current exception
            exc_value = ir.Constant(ir.IntType(8).as_pointer(), None)
        
        # Throw exception
        null = ir.Constant(ir.IntType(8).as_pointer(), None)
        self.builder.call(cxa_throw, [exc_value, null, null])
        
        # Add unreachable after throw
        self.builder.unreachable()
        return None
    
    def generate_with(self, with_node: 'IRWith') -> ir.Value:
        """
        Generate LLVM code for 'with' statement (context manager).
        
        Strategy:
        1. Call __enter__ method on context object
        2. Store result in target variable
        3. Execute body in try block
        4. Call __exit__ in finally block (always executes)
        """
        # Generate context manager expression
        context_mgr = self.generate_instruction(with_node.context_expr)
        
        # Create blocks
        enter_block = self.current_function.append_basic_block(name="with.enter")
        body_block = self.current_function.append_basic_block(name="with.body")
        exit_block = self.current_function.append_basic_block(name="with.exit")
        continue_block = self.current_function.append_basic_block(name="with.cont")
        
        # Enter: call __enter__
        self.builder.branch(enter_block)
        self.builder = ir.IRBuilder(enter_block)
        
        # In full implementation: call context_mgr.__enter__()
        enter_result = context_mgr if context_mgr else ir.Constant(ir.IntType(64), 0)  # Placeholder
        
        if with_node.var_name and enter_result:
            # Store __enter__ result in variable
            var_alloca = self.builder.alloca(enter_result.type, name=with_node.var_name)
            self.builder.store(enter_result, var_alloca)
            self.variables[with_node.var_name] = var_alloca
        
        self.builder.branch(body_block)
        
        # Body: execute with block body
        self.builder = ir.IRBuilder(body_block)
        for stmt in with_node.body:
            self.generate_instruction(stmt)
        
        self.builder.branch(exit_block)
        
        # Exit: call __exit__ (always executes, like finally)
        self.builder = ir.IRBuilder(exit_block)
        
        # In full implementation: call context_mgr.__exit__(exc_type, exc_val, exc_tb)
        # For now, just placeholder
        
        self.builder.branch(continue_block)
        self.builder = ir.IRBuilder(continue_block)
        
        return None

        pmb.populate(pm)
        
        # Run optimization passes
        pm.run(llvm_module)
        
        return str(llvm_module)
    
    def get_ir(self) -> str:
        """Get LLVM IR as string"""
        return str(self.module)
    
    # ========== Week 2: OOP Code Generation (Days 3-5) ==========
    
    def generate_new_object(self, instr: 'IRNewObject') -> ir.Value:
        """
        Generate object allocation and initialization (Week 2 Day 3)
        
        Creates:
        1. Allocate memory with malloc
        2. Cast to correct struct type
        3. Call __init__ if exists
        4. Return object pointer
        """
        class_name = instr.class_name
        
        # Get struct type
        if class_name not in self.class_types:
            raise ValueError(f"Class {class_name} not found")
        
        struct_type = self.class_types[class_name]
        
        # Calculate size of struct
        size = struct_type.get_abi_size(self.module.data_layout)
        size_const = ir.Constant(ir.IntType(64), size)
        
        # Call malloc
        raw_ptr = self.builder.call(self.malloc_func, [size_const], name=f"malloc_{class_name}")
        
        # Cast to struct pointer type
        struct_ptr_type = struct_type.as_pointer()
        obj_ptr = self.builder.bitcast(raw_ptr, struct_ptr_type, name=f"obj_{class_name}")
        
        # Call __init__ if it exists
        init_name = f"{class_name}___init__"
        try:
            init_func = self.module.get_global(init_name)
            # Generate arguments
            args = [obj_ptr]  # self parameter
            for arg in instr.args:
                args.append(self.generate_instruction(arg))
            self.builder.call(init_func, args)
        except KeyError:
            # No __init__ method, that's fine
            pass
        
        # Store in variable if result specified
        if instr.result:
            if instr.result.name not in self.variables:
                alloca = self.builder.alloca(struct_ptr_type, name=instr.result.name)
                self.variables[instr.result.name] = alloca
            self.builder.store(obj_ptr, self.variables[instr.result.name])
        
        return obj_ptr
    
    def generate_get_attr(self, instr: 'IRGetAttr') -> ir.Value:
        """
        Generate attribute access using GEP (Week 2 Day 4)
        
        obj.attribute → getelementptr + load
        """
        # Get object pointer
        obj = self.generate_instruction(instr.object)
        
        # Determine class name from object type
        # For now, extract from variable name or type
        class_name = None
        if hasattr(instr.object, 'name'):
            # Try to get class from variable type
            var_name = instr.object.name
            if var_name in self.variables:
                ptr_type = self.variables[var_name].type.pointee
                # Extract class name from struct type name
                if hasattr(ptr_type, 'name'):
                    class_name = ptr_type.name.replace('class.', '')
        
        if not class_name or class_name not in self.class_attr_names:
            # Fallback: return zero for now
            return ir.Constant(ir.IntType(64), 0)
        
        # Get attribute index
        attr_names = self.class_attr_names[class_name]
        if instr.attribute not in attr_names:
            return ir.Constant(ir.IntType(64), 0)
        
        attr_index = attr_names.index(instr.attribute)
        
        # Generate GEP to get attribute pointer
        zero = ir.Constant(ir.IntType(32), 0)
        index = ir.Constant(ir.IntType(32), attr_index)
        attr_ptr = self.builder.gep(obj, [zero, index], name=f"attr_{instr.attribute}_ptr")
        
        # Load value
        value = self.builder.load(attr_ptr, name=f"attr_{instr.attribute}")
        
        # Store in result variable if specified
        if instr.result and instr.result.name not in self.variables:
            result_type = self.type_to_llvm(instr.result.typ)
            alloca = self.builder.alloca(result_type, name=instr.result.name)
            self.variables[instr.result.name] = alloca
            self.builder.store(value, alloca)
        
        return value
    
    def generate_set_attr(self, instr: 'IRSetAttr') -> None:
        """
        Generate attribute assignment using GEP (Week 2 Day 4)
        
        obj.attribute = value → getelementptr + store
        """
        # Get object pointer
        obj = self.generate_instruction(instr.object)
        
        # Get value to store
        value = self.generate_instruction(instr.value)
        
        # Determine class name (same logic as get_attr)
        class_name = None
        if hasattr(instr.object, 'name'):
            var_name = instr.object.name
            if var_name in self.variables:
                ptr_type = self.variables[var_name].type.pointee
                if hasattr(ptr_type, 'name'):
                    class_name = ptr_type.name.replace('class.', '')
        
        if not class_name or class_name not in self.class_attr_names:
            return None
        
        # Get attribute index
        attr_names = self.class_attr_names[class_name]
        if instr.attribute not in attr_names:
            return None
        
        attr_index = attr_names.index(instr.attribute)
        
        # Generate GEP to get attribute pointer
        zero = ir.Constant(ir.IntType(32), 0)
        index = ir.Constant(ir.IntType(32), attr_index)
        attr_ptr = self.builder.gep(obj, [zero, index], name=f"attr_{instr.attribute}_ptr")
        
        # Store value
        self.builder.store(value, attr_ptr)
        
        return None
    
    def generate_method_call(self, instr: 'IRMethodCall') -> ir.Value:
        """
        Generate method call (Week 2 Day 5)
        
        obj.method(args) → call method_func(obj, args...)
        """
        # Get object pointer
        obj = self.generate_instruction(instr.object)
        
        # Determine class name
        class_name = None
        if hasattr(instr.object, 'name'):
            var_name = instr.object.name
            if var_name in self.variables:
                ptr_type = self.variables[var_name].type.pointee
                if hasattr(ptr_type, 'name'):
                    class_name = ptr_type.name.replace('class.', '')
        
        if not class_name:
            return ir.Constant(ir.IntType(64), 0)
        
        # Construct method name: ClassName_methodname
        method_name = f"{class_name}_{instr.method}"
        
        # Look up method function
        try:
            method_func = self.module.get_global(method_name)
        except KeyError:
            # Method not found
            return ir.Constant(ir.IntType(64), 0)
        
        # Build arguments: self + args
        args = [obj]
        for arg in instr.args:
            args.append(self.generate_instruction(arg))
        
        # Call method
        result = self.builder.call(method_func, args, name="method_call")
        
        # Store in result variable if specified
        if instr.result and instr.result.name not in self.variables:
            result_type = self.type_to_llvm(instr.result.typ)
            alloca = self.builder.alloca(result_type, name=instr.result.name)
            self.variables[instr.result.name] = alloca
            self.builder.store(result, alloca)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("LLVM IR GENERATION - Phase 1.3")
    print("=" * 80)
    
    # Test 1: Simple function
    print("\n--- Test 1: Generating LLVM IR for add function ---")
    
    # Create simple IR function
    func = IRFunction(
        name="add",
        param_names=["x", "y"],
        param_types=[Type(TypeKind.INT), Type(TypeKind.INT)],
        return_type=Type(TypeKind.INT)
    )
    
    entry = IRBasicBlock("entry")
    x_var = IRVar("x", Type(TypeKind.INT))
    y_var = IRVar("y", Type(TypeKind.INT))
    x_load = IRLoad(x_var)
    y_load = IRLoad(y_var)
    add_op = IRBinOp(IRNodeKind.ADD, x_load, y_load, Type(TypeKind.INT))
    ret = IRReturn(add_op)
    
    entry.add_instruction(add_op)
    entry.add_instruction(ret)
    func.add_block(entry)
    
    # Generate LLVM IR
    module = IRModule("test")
    module.add_function(func)
    
    codegen = LLVMCodeGen()
    llvm_ir = codegen.generate_module(module)
    
    print(llvm_ir)
    
    # Test 2: Function with control flow
    print("\n" + "=" * 80)
    print("--- Test 2: Function with if statement ---")
    print("=" * 80)
    
    max_func = IRFunction(
        name="max_val",
        param_names=["a", "b"],
        param_types=[Type(TypeKind.INT), Type(TypeKind.INT)],
        return_type=Type(TypeKind.INT)
    )
    
    # Entry block
    entry2 = IRBasicBlock("entry")
    a_var = IRVar("a", Type(TypeKind.INT))
    b_var = IRVar("b", Type(TypeKind.INT))
    a_load = IRLoad(a_var)
    b_load = IRLoad(b_var)
    cmp = IRBinOp(IRNodeKind.GT, a_load, b_load, Type(TypeKind.BOOL))
    branch = IRBranch(cmp, "then", "else")
    
    entry2.add_instruction(cmp)
    entry2.add_instruction(branch)
    
    # Then block
    then_block = IRBasicBlock("then")
    a_load2 = IRLoad(a_var)
    then_block.add_instruction(IRReturn(a_load2))
    
    # Else block
    else_block = IRBasicBlock("else")
    b_load2 = IRLoad(b_var)
    else_block.add_instruction(IRReturn(b_load2))
    
    max_func.add_block(entry2)
    max_func.add_block(then_block)
    max_func.add_block(else_block)
    
    module2 = IRModule("test2")
    module2.add_function(max_func)
    
    codegen2 = LLVMCodeGen()
    llvm_ir2 = codegen2.generate_module(module2)
    
    print(llvm_ir2)
    
    print("\n" + "=" * 80)
    print("✅ Phase 1.3 LLVM Code Generation Complete!")
    print("=" * 80)
    print("\nNext: Phase 1.3 - Native Code Compilation")
