from functools import singledispatchmethod

import javalang.tree


class StatementConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, statement):
        return ""

    @convert.register(javalang.tree.BlockStatement)
    @classmethod
    def __convert_block_statement_to_cfg(cls, block_statement):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(block_statement.statements)

    @classmethod
    def __convert_condition_and_body_to_cfg(cls, statement):
        from .CfgConvertor import CfgConvertor
        from .CfgNode import CfgNode
        condition = CfgConvertor.convert(statement.condition)
        condition = CfgNode(condition)
        body = CfgConvertor.convert(statement.body)
        body_entry = CfgNode()
        body_exit = CfgNode()
        if isinstance(body, tuple):
            body_entry, body_exit = body
        else:
            body_node = CfgNode(body)
            body_entry.add(body_node)
            body_node.add(body_exit)
        return condition, body_entry, body_exit

    @convert.register(javalang.tree.DoStatement)
    @classmethod
    def __convert_do_statement_to_cfg(cls, do_statement):
        from .CfgNode import CfgNode
        condition, body_entry, body_exit = cls.__convert_condition_and_body_to_cfg(do_statement)
        condition.source_code = " while ( " + condition.source_code + " ) "
        entry_node = CfgNode()
        exit_node = CfgNode()
        body_exit.add(condition)
        condition.add(body_entry)

        entry_node.add(body_entry)
        condition.add(exit_node)
        return entry_node, exit_node

    @convert.register(javalang.tree.ForStatement)
    @classmethod
    def __convert_for_statement_to_cfg(cls, for_statement):
        from .CfgConvertor import CfgConvertor
        from .CfgNode import CfgNode
        control = CfgConvertor.convert(for_statement.control)
        control = CfgNode(control)
        body_cfg = CfgConvertor.convert(for_statement.body)
        body_entry = CfgNode()
        body_exit = CfgNode()
        if isinstance(body_cfg, tuple):
            body_entry, body_exit = body_cfg
        else:
            body_cfg = CfgNode(body_cfg)
            body_entry.add(body_cfg)
            body_cfg.add(body_exit)
        entry_node = CfgNode()
        exit_node = CfgNode()

        body_exit.add(control)
        control.add(body_entry)

        entry_node.add(control)
        body_exit.add(exit_node)
        return entry_node, exit_node

    @convert.register(javalang.tree.IfStatement)
    @classmethod
    def __convert_if_statement_to_cfg(cls, if_statement):
        from .CfgConvertor import CfgConvertor
        from .CfgNode import CfgNode
        condition = CfgConvertor.convert(if_statement.condition)
        condition = " if ( " + condition + " ) "
        condition = CfgNode(condition)

        then_statement_entry = CfgNode()
        then_statement_exit = CfgNode()
        then_statement = CfgConvertor.convert(if_statement.then_statement)
        if isinstance(then_statement, tuple):
            then_statement_entry, then_statement_exit = then_statement
        else:
            then_statement = CfgNode(then_statement)
            then_statement_entry.add(then_statement)
            then_statement.add(then_statement_exit)

        else_statement_entry = CfgNode()
        else_statement_exit = CfgNode()
        else_statement = CfgConvertor.convert(if_statement.else_statement)
        if isinstance(else_statement, tuple):
            else_statement_entry, else_statement_exit = else_statement
        else:
            else_statement = CfgNode(else_statement)
            else_statement_entry.add(else_statement)
            else_statement.add(else_statement_exit)

        entry_node = CfgNode()
        exit_node = CfgNode()

        entry_node.add(condition)
        condition.add(then_statement_entry)
        condition.add(else_statement_entry)
        then_statement_exit.add(exit_node)
        else_statement_exit.add(exit_node)
        return entry_node, exit_node

    @convert.register(javalang.tree.ReturnStatement)
    @classmethod
    def __convert_return_statement_to_cfg(cls, return_statement):
        from .CfgConvertor import CfgConvertor
        from .CfgNode import CfgNode
        expression = CfgConvertor.convert(return_statement.expression)
        expression = " return " + expression
        return_node = CfgNode(expression)
        entry_node = CfgNode()
        return_node.is_return = True
        entry_node.add(return_node)
        return entry_node, return_node

    @convert.register(javalang.tree.StatementExpression)
    @classmethod
    def __convert_statement_expression_to_string(cls, statement_expression):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(statement_expression.expression)

    @convert.register(javalang.tree.SwitchStatement)
    @classmethod
    def __convert_switch_statement_to_cfg(cls, switch_statement):
        from .CfgConvertor import CfgConvertor
        from .CfgNode import CfgNode
        expression = CfgConvertor.convert(switch_statement.expression)
        expression = " switch ( " + expression + " ) "
        expression = CfgNode(expression)
        switch_case_list = []
        for switch_case in switch_statement.cases:
            switch_case_list.append(CfgConvertor.convert(switch_case))

        entry_node = CfgNode()
        exit_node = CfgNode()
        entry_node.add(expression)
        for switch_case in switch_case_list:
            switch_case_entry_node, switch_case_exit_node = switch_case
            expression.add(switch_case_entry_node)
            switch_case_exit_node.add(exit_node)
        return entry_node, exit_node

    @convert.register(javalang.tree.SynchronizedStatement)
    @classmethod
    def __convert_synchronize_statement_to_cfg(cls, synchronized_statement):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(synchronized_statement.block)

    @convert.register(javalang.tree.ThrowStatement)
    @classmethod
    def __convert_throw_statement_to_cfg(cls, throw_statement):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(throw_statement.expression)

    @convert.register(javalang.tree.TryStatement)
    @classmethod
    def __convert_try_statement_to_cfg(cls, try_statement):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(try_statement.block)

    @convert.register(javalang.tree.WhileStatement)
    @classmethod
    def __convert_while_statement_to_cfg(cls, while_statement):
        from .CfgNode import CfgNode
        condition, body_entry, body_exit = cls.__convert_condition_and_body_to_cfg(while_statement)

        condition.source_code = "while ( " + condition.source_code + " )"
        entry_node = CfgNode()
        exit_node = CfgNode()
        body_exit.add(condition)
        condition.add(body_entry)

        entry_node.add(condition)
        body_exit.add(exit_node)
        return entry_node, exit_node
