class FuncNameExtractor:
    @classmethod
    def extract(cls, func_name):
        try:
            from .Utility import Utility
            func_name = Utility.filter_func_name(func_name)
            func_name = Utility.tokenize_stop_words_removal_and_lemmatization(func_name)
        except Exception as e:
            func_name = ''
            print(e)
        return func_name
