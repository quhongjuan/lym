from functools import singledispatchmethod

import javalang.tree

from .CfgConvertor import CfgConvertor


class CreatorConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, creator):
        if creator is not None:
            return "new " + CfgConvertor.convert(creator.type)
        return ""

    @convert.register(javalang.tree.ClassCreator)
    @classmethod
    def __convert_class_creator_to_string(cls, class_creator):
        return "new " + CfgConvertor.convert(class_creator.type) + " ( " + CfgConvertor.convert_list_to_string(
            class_creator.constructor_type_arguments) + CfgConvertor.convert_list_to_string(
            class_creator.arguments) + " ) "

    @convert.register(javalang.tree.ArrayCreator)
    @classmethod
    def __convert_array_creator_to_string(cls, array_creator):
        return "new " + CfgConvertor.convert(array_creator.type) + " [ " + CfgConvertor.convert_list_to_string(
            array_creator.dimensions) + " ] "
