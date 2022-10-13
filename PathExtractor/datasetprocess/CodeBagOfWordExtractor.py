class CodeBagOfWordExtractor:
    @classmethod
    def extract(cls, code):
        try:
            from .Utility import Utility
            code = Utility.filter_code(code)
            code = Utility.tokenize_stop_words_removal_and_lemmatization(code)
        except Exception as e:
            code = ''
            print(e)
        return code
