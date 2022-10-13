class DocStringExtractor:
    @classmethod
    def extract(cls, docstring):
        try:
            from .Utility import Utility
            docstring = Utility.filter_docstring(docstring)
            docstring = Utility.tokenize_stop_words_removal_and_lemmatization(docstring)
        except Exception as e:
            docstring = ''
            print(e)
        return docstring
