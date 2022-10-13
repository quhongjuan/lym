from functools import singledispatchmethod


class SwitchStatementCaseConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, switch_statement_case):
        from .CfgConvertor import CfgConvertor
        from .CfgNode import CfgNode
        case_condition = CfgConvertor.convert_list_to_string(switch_statement_case.case)
        case_condition = " case " + case_condition
        case_condition = CfgNode(case_condition)
        statements_entry, statements_exit = CfgConvertor.convert(switch_statement_case.statements)

        entry_node = CfgNode()
        exit_node = CfgNode()

        entry_node.add(case_condition)
        case_condition.add(statements_entry)
        statements_exit.add(exit_node)
        return entry_node, exit_node
