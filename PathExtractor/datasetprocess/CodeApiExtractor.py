import javalang
import javalang.tree


class CodeApiExtractor:
    @classmethod
    def __check_arguments(cls, node, identifier_filter):
        arguments_api_sequence = []
        for argument in node.arguments:
            if isinstance(argument, javalang.tree.MethodInvocation):
                api = [identifier_filter.get(argument.qualifier, argument.qualifier), argument.member]
                arguments_api_sequence.append(api)
                arguments_api_sequence.extend(cls.__check_arguments(argument, identifier_filter))
            if isinstance(argument, javalang.tree.ClassCreator):
                api = [argument.type.name, 'new']
                arguments_api_sequence.append(api)
                arguments_api_sequence.extend(cls.__check_arguments(argument, identifier_filter))
        return arguments_api_sequence

    @classmethod
    def __check_selectors(cls, node, identifier_filter):
        selectors_api_sequence = []
        if node.selectors is not None:
            for selector in node.selectors:
                if isinstance(selector, javalang.tree.MethodInvocation):
                    if node.qualifier is None:
                        selectors_api_sequence.append([node.type.name, selector.member])
                    else:
                        selectors_api_sequence.append(
                            [identifier_filter.get(node.qualifier, node.qualifier), selector.member])
        return selectors_api_sequence

    @classmethod
    def extract(cls, method_source_code):
        code = "package temp; class Temp {%s}" % method_source_code
        api_sequences = []
        try:
            tree = javalang.parse.parse(code)
        except Exception as e:
            print(e)
            tree = ""
        identifier_filter = {}
        for _, node in tree:
            if isinstance(node, javalang.tree.FormalParameter):
                identifier_filter[node.name] = node.type.name
            elif isinstance(node, javalang.tree.LocalVariableDeclaration):
                for declarator in node.declarators:
                    identifier_filter[declarator.name] = node.type.name
            elif isinstance(node, javalang.tree.ClassCreator):
                api = [node.type.name, 'new']
                api_sequences.append(api)
                api_sequences.extend(cls.__check_selectors(node, identifier_filter))
            elif isinstance(node, javalang.tree.MethodInvocation):
                if node.qualifier == '':
                    continue
                if node.qualifier is None:
                    if len(api_sequences) == 0:
                        continue
                    node.qualifier = api_sequences[-1][0]
                sub_api_sequences = cls.__check_arguments(node, identifier_filter)
                sub_api_sequences.append([identifier_filter.get(node.qualifier, node.qualifier), node.member])
                api_sequences.extend(sub_api_sequences)
                api_sequences.extend(cls.__check_selectors(node, identifier_filter))
        api_sequences = ['.'.join(item) for item in api_sequences]
        return ' '.join(api_sequences)
