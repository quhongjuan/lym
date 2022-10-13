from functools import singledispatchmethod

import javalang.tree


class TypeConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, type_node):
        if type_node is not None:
            return type_node.name
        return ""

    @convert.register(javalang.tree.ReferenceType)
    @classmethod
    def convert_reference_type_to_string(cls, reference_type):
        from .CfgConvertor import CfgConvertor
        sub_type = CfgConvertor.convert(reference_type.sub_type)
        if sub_type != '':
            return reference_type.name + '.' + sub_type
        return reference_type.name
