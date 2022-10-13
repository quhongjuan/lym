from functools import singledispatchmethod


class VariableDeclaratorConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, variable_declarator):
        from .CfgConvertor import CfgConvertor
        initializer = CfgConvertor.convert(variable_declarator.initializer)
        if initializer != "":
            return variable_declarator.name + " = " + initializer
        return variable_declarator.name
