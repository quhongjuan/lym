from functools import singledispatchmethod

import javalang.tree


class ExpressionConvertor(object):
    @singledispatchmethod
    @classmethod
    def convert(cls, expression):
        return ""

    @convert.register(javalang.tree.ArraySelector)
    @classmethod
    def __convert_array_selector_to_string(cls, array_selector):
        from .CfgConvertor import CfgConvertor
        return "[ " + CfgConvertor.convert(array_selector.index) + " ]"

    @convert.register(javalang.tree.Assignment)
    @classmethod
    def __convert_assignment_to_string(cls, assignment):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(assignment.expressionl) + " " + assignment.type + " " + CfgConvertor.convert(
            assignment.value)

    @convert.register(javalang.tree.BinaryOperation)
    @classmethod
    def __convert_binary_operation_to_string(cls, binary_operation):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(
            binary_operation.operandl) + " " + binary_operation.operator + " " + CfgConvertor.convert(
            binary_operation.operandr)

    @classmethod
    def __get_path_length(cls, path_length):
        return len(path_length)

    @convert.register(javalang.tree.LambdaExpression)
    @classmethod
    def __convert_lambda_expression_to_string(cls, lambda_expression):
        from .CfgConvertor import CfgConvertor
        lambda_body = CfgConvertor.convert(lambda_expression.body)
        if isinstance(lambda_body, tuple):
            lambda_body_entry, lambda_body_exit = lambda_body
            lambda_body_paths = CfgConvertor.convert_cfg_to_paths(lambda_body_entry)
            if len(lambda_body_paths) == 0:
                lambda_body = ""
            else:
                lambda_body_paths.sort(key=cls.__get_path_length)
                lambda_body = ' '.join(lambda_body_paths[0])
        return CfgConvertor.convert_list_to_string(lambda_expression.parameters) + lambda_body

    @convert.register(javalang.tree.MethodReference)
    @classmethod
    def __convert_method_reference_to_string(cls, method_reference):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(method_reference.method) + " " + CfgConvertor.convert(
            method_reference.expression) + " " + CfgConvertor.convert_list_to_string(method_reference.type_arguments)

    @convert.register(javalang.tree.Primary)
    @classmethod
    def __convert_primary_to_string(cls, primary):
        from .PrimaryConvertor import PrimaryConvertor
        from .CfgConvertor import CfgConvertor

        selectors = ""
        if primary.selectors is not None and len(primary.selectors) != 0:
            for selector in primary.selectors:
                if isinstance(selector, javalang.tree.ArraySelector):
                    selectors = selectors + " " + CfgConvertor.convert(selector)
                else:
                    selectors = selectors + "." + CfgConvertor.convert(selector)
        prefix_operators = ""
        if primary.prefix_operators is not None and len(primary.prefix_operators) != 0:
            for prefix_operator in primary.prefix_operators:
                prefix_operators = prefix_operators + prefix_operator
        postfix_operators = ""
        if primary.postfix_operators is not None and len(primary.postfix_operators) != 0:
            for postfix_operator in primary.postfix_operators:
                postfix_operators = postfix_operators + postfix_operator
        qualifier = CfgConvertor.convert(primary.qualifier)
        if qualifier != "":
            qualifier = qualifier + "."
        return prefix_operators + " " + qualifier + PrimaryConvertor.convert(
            primary) + " " + postfix_operators + " " + selectors

    @convert.register(javalang.tree.TernaryExpression)
    @classmethod
    def __convert_ternary_expression_to_string(cls, ternary_expression):
        from .CfgConvertor import CfgConvertor
        return CfgConvertor.convert(ternary_expression.condition) + "?" + CfgConvertor.convert(
            ternary_expression.if_true) + ":" + CfgConvertor.convert(ternary_expression.if_false)
