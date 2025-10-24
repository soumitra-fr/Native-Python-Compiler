"""
Decorator Support for Native Python Compiler
Week 5: Implement @property, @staticmethod, @classmethod

Handles:
- @property decorator → getter methods
- @property.setter decorator → setter methods
- @property.deleter decorator → deleter methods
- @staticmethod decorator → static methods (no self)
- @classmethod decorator → class methods (cls parameter)
"""

import ast
from typing import Optional, List, Dict
from compiler.ir.ir_nodes import IRFunction, IRClass


class DecoratorHandler:
    """
    Handles decorator processing during AST lowering
    
    Transforms decorated methods into appropriate IR representations
    """
    
    def __init__(self):
        self.properties: Dict[str, Dict[str, IRFunction]] = {}  # {prop_name: {get/set/del: func}}
        
    def is_property_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is @property"""
        if isinstance(decorator, ast.Name):
            return decorator.id == 'property'
        return False
    
    def is_property_setter(self, decorator: ast.expr) -> Optional[str]:
        """Check if decorator is @prop.setter, return property name"""
        if isinstance(decorator, ast.Attribute):
            if decorator.attr == 'setter' and isinstance(decorator.value, ast.Name):
                return decorator.value.id
        return None
    
    def is_property_deleter(self, decorator: ast.expr) -> Optional[str]:
        """Check if decorator is @prop.deleter, return property name"""
        if isinstance(decorator, ast.Attribute):
            if decorator.attr == 'deleter' and isinstance(decorator.value, ast.Name):
                return decorator.value.id
        return None
    
    def is_staticmethod_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is @staticmethod"""
        if isinstance(decorator, ast.Name):
            return decorator.id == 'staticmethod'
        return False
    
    def is_classmethod_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is @classmethod"""
        if isinstance(decorator, ast.Name):
            return decorator.id == 'classmethod'
        return False
    
    def process_decorators(self, node: ast.FunctionDef) -> Dict[str, any]:
        """
        Process decorators on a function
        
        Returns:
            Dictionary with decorator info:
            {
                'is_property': bool,
                'property_type': 'getter'|'setter'|'deleter',
                'property_name': str,
                'is_staticmethod': bool,
                'is_classmethod': bool
            }
        """
        info = {
            'is_property': False,
            'property_type': None,
            'property_name': None,
            'is_staticmethod': False,
            'is_classmethod': False
        }
        
        for decorator in node.decorator_list:
            # @property
            if self.is_property_decorator(decorator):
                info['is_property'] = True
                info['property_type'] = 'getter'
                info['property_name'] = node.name
            
            # @prop.setter
            setter_name = self.is_property_setter(decorator)
            if setter_name:
                info['is_property'] = True
                info['property_type'] = 'setter'
                info['property_name'] = setter_name
            
            # @prop.deleter
            deleter_name = self.is_property_deleter(decorator)
            if deleter_name:
                info['is_property'] = True
                info['property_type'] = 'deleter'
                info['property_name'] = deleter_name
            
            # @staticmethod
            if self.is_staticmethod_decorator(decorator):
                info['is_staticmethod'] = True
            
            # @classmethod
            if self.is_classmethod_decorator(decorator):
                info['is_classmethod'] = True
        
        return info
    
    def register_property(self, prop_name: str, prop_type: str, func: IRFunction):
        """Register a property getter/setter/deleter"""
        if prop_name not in self.properties:
            self.properties[prop_name] = {}
        self.properties[prop_name][prop_type] = func
    
    def get_property_getter(self, prop_name: str) -> Optional[IRFunction]:
        """Get getter function for a property"""
        if prop_name in self.properties:
            return self.properties[prop_name].get('getter')
        return None
    
    def get_property_setter(self, prop_name: str) -> Optional[IRFunction]:
        """Get setter function for a property"""
        if prop_name in self.properties:
            return self.properties[prop_name].get('setter')
        return None
    
    def transform_property_access(self, obj_name: str, attr_name: str, 
                                  is_load: bool) -> Optional[str]:
        """
        Transform property access into method call
        
        Args:
            obj_name: Name of object
            attr_name: Name of attribute/property
            is_load: True for obj.prop (get), False for obj.prop = val (set)
        
        Returns:
            Method call string to use, or None if not a property
        """
        if attr_name in self.properties:
            if is_load:
                # obj.prop → obj.get_prop()
                return f"{obj_name}.get_{attr_name}()"
            else:
                # obj.prop = val → obj.set_prop(val)
                return f"{obj_name}.set_{attr_name}"
        return None


def create_property_descriptor(prop_name: str, getter: Optional[IRFunction],
                               setter: Optional[IRFunction],
                               deleter: Optional[IRFunction]) -> dict:
    """
    Create property descriptor for IR
    
    Args:
        prop_name: Property name
        getter: Getter function
        setter: Setter function
        deleter: Deleter function
    
    Returns:
        Property descriptor dictionary
    """
    return {
        'name': prop_name,
        'type': 'property',
        'getter': getter,
        'setter': setter,
        'deleter': deleter,
        'fget': getter.name if getter else None,
        'fset': setter.name if setter else None,
        'fdel': deleter.name if deleter else None
    }


def is_decorated_method(node: ast.FunctionDef) -> bool:
    """Check if function has any decorators"""
    return len(node.decorator_list) > 0


def get_decorator_names(node: ast.FunctionDef) -> List[str]:
    """Get list of decorator names"""
    names = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            names.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.append(f"{decorator.value.id}.{decorator.attr}")
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                names.append(decorator.func.id)
    return names


# Global decorator handler
_decorator_handler = DecoratorHandler()


def get_decorator_handler() -> DecoratorHandler:
    """Get global decorator handler instance"""
    return _decorator_handler


def reset_decorator_handler():
    """Reset decorator handler (for testing)"""
    global _decorator_handler
    _decorator_handler = DecoratorHandler()
