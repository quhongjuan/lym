from functools import singledispatchmethod


class ForControlConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, for_control):
        from .CfgConvertor import CfgConvertor
        updates = CfgConvertor.convert_list_to_string(for_control.update)
        inits = ""
        if isinstance(for_control.init, list):
            inits = CfgConvertor.convert_list_to_string(for_control.init)
        else:
            inits = CfgConvertor.convert(list)
        return "for ( " + inits + " ; " + CfgConvertor.convert(
            for_control.condition) + " ; " + updates + " ) "
