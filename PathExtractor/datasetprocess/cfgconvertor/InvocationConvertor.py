from functools import singledispatchmethod

import javalang.tree


class InvocationConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, invocation):
        return ""

    @convert.register(javalang.tree.MethodInvocation)
    @classmethod
    def __convert_method_invocation_to_string(cls, method_invocation):
        from .CfgConvertor import CfgConvertor
        return method_invocation.member + " ( " + CfgConvertor.convert_list_to_string(
            method_invocation.arguments) + " ) "
