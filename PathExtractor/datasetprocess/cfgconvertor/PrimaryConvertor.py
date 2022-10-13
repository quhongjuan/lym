from functools import singledispatchmethod

import javalang.tree


class PrimaryConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, primary):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert_list_to_string(primary.selectors)

    @convert.register(javalang.tree.ClassReference)
    @classmethod
    def __convert_class_reference_to_string(cls, class_reference):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(class_reference.type) + " " + CfgConvertor.convert_list_to_string(
            class_reference.selectors)

    @convert.register(javalang.tree.Creator)
    @classmethod
    def __convert_creator_to_string(cls, creator):
        from .CreatorConvertor import CreatorConvertor
        return CreatorConvertor.convert(creator)

    @convert.register(javalang.tree.Invocation)
    @classmethod
    def __convert_invocation_to_string(cls, invocation):
        from .InvocationConvertor import InvocationConvertor
        return InvocationConvertor.convert(invocation)

    @convert.register(javalang.tree.Literal)
    @classmethod
    def __convert_literal_to_string(cls, literal):
        return literal.value

    @convert.register(javalang.tree.MemberReference)
    @classmethod
    def __convert_member_reference_to_string(cls, member_reference):
        return member_reference.member
