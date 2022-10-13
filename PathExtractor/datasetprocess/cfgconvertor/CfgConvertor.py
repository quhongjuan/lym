import re
from functools import singledispatchmethod

import javalang
from javalang.tokenizer import Null
import javalang.tree


class CfgConvertor(object):
    """提取源代码的控制流图，考虑for、while、do、if、switch、try及相互嵌套的控制流
    """

    @singledispatchmethod
    @classmethod
    def convert(cls, node):
        return ""

    @convert.register(javalang.tree.Declaration)
    @classmethod
    def __convert_declaration_to_string(cls, declaration):
        from .DeclarationConvertor import DeclarationConvertor
        return DeclarationConvertor.convert(declaration)

    @convert.register(javalang.tree.EnhancedForControl)
    @classmethod
    def __convert_enhanced_for_control_to_string(cls, enhanced_for_control):
        from .EnhancedForControlConvertor import EnhancedForControlConvertor
        return EnhancedForControlConvertor.convert(enhanced_for_control)

    @convert.register(javalang.tree.Expression)
    @classmethod
    def __convert_expression_to_string(cls, expression):
        from .ExpressionConvertor import ExpressionConvertor
        return ExpressionConvertor.convert(expression)

    @convert.register(javalang.tree.ForControl)
    @classmethod
    def __convert_for_control_to_string(cls, for_control):
        from .ForControlConvertor import ForControlConvertor
        return ForControlConvertor.convert(for_control)

    @convert.register(list)
    @classmethod
    def __convert_list_to_cfg(cls, node_list):
        from .CfgNode import CfgNode
        entry_node = CfgNode()
        exit_node = CfgNode()
        if node_list is None or len(node_list) == 0:
            entry_node.add(exit_node)
            return entry_node, exit_node

        temp_node = entry_node
        for item in node_list:
            converted = cls.convert(item)
            item_entry = CfgNode()
            item_exit = CfgNode()
            if isinstance(converted, tuple):
                item_entry, item_exit = converted
                temp_node.add(item_entry)
                temp_node = item_exit
            else:
                converted_node = CfgNode(converted)
                item_entry.add(converted_node)
                converted_node.add(item_exit)
                temp_node.add(item_entry)
                temp_node = item_exit

        temp_node.add(exit_node)
        return entry_node, exit_node

    @classmethod
    def convert_list_to_string(cls, item_list):
        """工具函数，将列表中的每个元素转为string，再拼接
        """
        result = ""
        if item_list is not None and len(item_list) != 0:
            result_list = []
            for item in item_list:
                result_list.append(CfgConvertor.convert(item))

            for item in result_list[:-1]:
                result = result + item + " , "

            result = result + result_list[-1]
        return result

    @classmethod
    def __simplify_cfg(cls, node):
        if node.visited:
            return
        node.visited = True
        if node.source_code == '':
            parent = node.parent
            children = node.children
            for p in parent:
                p.children.remove(node)
            for c in children:
                c.parent.remove(node)
            for p in parent:
                for c in children:
                    p.add(c)
        children = node.children.copy()
        for c in children:
            cls.__simplify_cfg(c)

    @classmethod
    def convert_method_source_code_to_cfg(cls, method_source_code):
        method_source_code = cls.__filter_source_code(method_source_code)
        method_source_code = "{%s}" % method_source_code
        tokens = javalang.tokenizer.tokenize(method_source_code)
        try:
            parser = javalang.parse.Parser(tokens)
            ast_tree = parser.parse_class_body()
        except Exception as e:
            print(e)
            from .CfgNode import CfgNode
            return CfgNode()
        method_node = ast_tree[0]
        from .DeclarationConvertor import DeclarationConvertor
        cfg = DeclarationConvertor.convert(method_node)
        entry_node, exit_node = cfg
        children = entry_node.children.copy()
        for node in children:
            cls.__simplify_cfg(node)
        return entry_node, method_node.name

    @convert.register(javalang.tree.Statement)
    @classmethod
    def __convert_statement_to_cfg(cls, statement):
        from .StatementConvertor import StatementConvertor
        return StatementConvertor.convert(statement)

    @convert.register(javalang.tree.SwitchStatementCase)
    @classmethod
    def __convert_switch_statement_case_to_cfg(cls, switch_statement_case):
        from .SwitchStatementCaseConvertor import SwitchStatementCaseConvertor
        return SwitchStatementCaseConvertor.convert(switch_statement_case)

    @convert.register(javalang.tree.Type)
    @classmethod
    def __convert_type_to_string(cls, type_node):
        from .TypeConvertor import TypeConvertor
        return TypeConvertor.convert(type_node)

    @convert.register(javalang.tree.VariableDeclarator)
    @classmethod
    def __convert_variable_declarator_to_string(cls, variable_declarator):
        from .VariableDeclaratorConvertor import VariableDeclaratorConvertor
        return VariableDeclaratorConvertor.convert(variable_declarator)

    @convert.register(str)
    @classmethod
    def __convert_str(cls, string):
        return string

    @classmethod
    def __filter_source_code(cls, source_code):
        source_code = re.sub(r'//.*', ' ', source_code)
        source_code = re.sub(r'/\*.*?\*/', ' ', source_code, flags=re.DOTALL)
        source_code = re.sub(r'[\n\r]', ' ', source_code)
        source_code = re.sub(r' {2,}', ' ', source_code)
        return source_code.strip()

    @classmethod
    def convert_class_source_code_to_cfg(cls, class_source_code):
        class_source_code = cls.__filter_source_code(class_source_code)
        class_source_code = class_source_code
        ast_tree = javalang.parse.parse(class_source_code)
        method_node = ast_tree.types[0].body[0]
        from .DeclarationConvertor import DeclarationConvertor
        cfg = DeclarationConvertor.convert(method_node)
        entry_node, exit_node = cfg
        children = entry_node.children.copy()
        for node in children:
            cls.__simplify_cfg(node)
        return entry_node

    @classmethod
    def convert_cfg_to_paths(cls, cfg_entry, method_name=Null):
        all_cannot_forward_paths = []
        for node in cfg_entry.children:
            if method_name == Null:
                current_path = [node.source_code]
            else:
                current_path = [method_name, node.source_code]
            current_visit = {node}
            cls.__get_path(node, current_visit, current_path, all_cannot_forward_paths)
            if len(all_cannot_forward_paths) >= 10:
                break
        return all_cannot_forward_paths

    @classmethod
    def __get_path(cls, node, current_visit, current_path, all_cannot_forward_paths):
        if len(all_cannot_forward_paths) >= 10:
            return
        if len(node.children) == 0:
            all_cannot_forward_paths.append(current_path.copy())
            return
        current_path_added = False
        for child in node.children:
            if child in current_visit and not current_path_added:
                all_cannot_forward_paths.append(current_path.copy())
                current_path_added = True
            elif child not in current_visit:
                current_path.append(child.source_code)
                current_visit.add(child)
                cls.__get_path(child, current_visit, current_path, all_cannot_forward_paths)
                if len(all_cannot_forward_paths) >= 10:
                    return
                current_path.pop()
                current_visit.remove(child)
