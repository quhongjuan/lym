from functools import singledispatchmethod


class EnhancedForControlConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, enhanced_for_control):
        from .CfgConvertor import CfgConvertor
        return "for ( " + CfgConvertor.convert(enhanced_for_control.var) + " : " + CfgConvertor.convert(
            enhanced_for_control.iterable) + " )"
