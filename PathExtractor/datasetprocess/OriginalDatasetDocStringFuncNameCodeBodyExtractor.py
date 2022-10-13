import json


class OriginalDatasetDocStringFuncNameCodeBodyExtractor:
    @classmethod
    def extract(cls, input_file, output_file):
        file_input = open(input_file, 'r', encoding='utf-8')
        file_output = open(output_file, 'w', encoding='utf-8')
        id_num = 1
        for line in file_input.readlines():
            json_line = json.loads(line.strip())
            docstring = json_line['docstring']
            func_name = json_line['func_name']
            code = json_line['code']
            extracted_data = {'ID': id_num, 'docstring': docstring, 'func_name': func_name, 'code': code}
            file_output.write(json.dumps(extracted_data) + '\n')
            id_num = id_num + 1
