from functools import singledispatchmethod

import javalang.tree


class DeclarationConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, declaration):
        return ""

    @convert.register(javalang.tree.FormalParameter)
    @classmethod
    def __convert_formal_parameter_to_string(cls, formal_parameter):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(formal_parameter.type) + " " + formal_parameter.name

    @convert.register(javalang.tree.ConstructorDeclaration)
    @classmethod
    def __convert_constructor_declaration_to_cfg(cls, constructor_declaration):
        from .CfgConvertor import CfgConvertor
        parameters = CfgConvertor.convert_list_to_string(constructor_declaration.parameters)

        from .CfgNode import CfgNode
        entry_node = CfgNode()
        exit_node = CfgNode()
        parameters_node = CfgNode(parameters)

        method_declaration_body_entry, method_declaration_body_exit = CfgConvertor.convert(constructor_declaration.body)

        entry_node.add(parameters_node)
        parameters_node.add(method_declaration_body_entry)
        method_declaration_body_exit.add(exit_node)

        return entry_node, exit_node

    @convert.register(javalang.tree.MethodDeclaration)
    @classmethod
    def __convert_method_declaration_to_cfg(cls, method_declaration):
        from .CfgConvertor import CfgConvertor
        parameters = CfgConvertor.convert_list_to_string(method_declaration.parameters)

        from .CfgNode import CfgNode
        entry_node = CfgNode()
        exit_node = CfgNode()
        parameters_node = CfgNode(parameters)

        method_declaration_body_entry, method_declaration_body_exit = CfgConvertor.convert(method_declaration.body)

        entry_node.add(parameters_node)
        parameters_node.add(method_declaration_body_entry)
        method_declaration_body_exit.add(exit_node)

        return entry_node, exit_node

    @convert.register(javalang.tree.VariableDeclaration)
    @classmethod
    def __convert_variable_declaration_to_string(cls, variable_declaration):
        from .CreatorConvertor import CfgConvertor
        declarators = CfgConvertor.convert_list_to_string(variable_declaration.declarators)
        return CfgConvertor.convert(variable_declaration.type) + " " + declarators
