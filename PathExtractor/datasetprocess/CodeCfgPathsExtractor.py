class CodeCfgPathsExtractor:
    @classmethod
    def extract(cls, method_source_code, need_method_name):
        try:
            from .cfgconvertor.CfgConvertor import CfgConvertor
            cfg_entry, method_name = CfgConvertor.convert_method_source_code_to_cfg(method_source_code)
            if need_method_name:
                paths = CfgConvertor.convert_cfg_to_paths(cfg_entry, method_name)
            else:
                paths = CfgConvertor.convert_cfg_to_paths(cfg_entry)
        except Exception as e:
            print(e)
            paths = []
        return paths
