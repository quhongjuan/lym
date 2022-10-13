import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class Utility:
    __lemmatizer = WordNetLemmatizer()
    __stop_words = set(stopwords.words('english'))
    __tag_dict = {"J": wordnet.ADJ,
                  "N": wordnet.NOUN,
                  "V": wordnet.VERB,
                  "R": wordnet.ADV}

    @classmethod
    def filter_code(cls, code):
        # remove line delimiter
        code = re.sub(r'[\n\r]', ' ', code)
        # retain only alphabet, digits, and, or, not, + - * / % = < > ?, this symbol may have meanings
        # and "-" may be prefix symbol or just connect between alphabet
        code = re.sub(r'[^A-Za-z0-9&|!+\-*/%=<>?]', ' ', code)
        # remove 2 or more continuous space
        code = re.sub(r' {2,}', ' ', code)
        # split word with java camel naming convention
        code = re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', code)
        code = code.lower()
        return code

    @classmethod
    def filter_docstring(cls, docstring):
        # remove line delimiter
        docstring = re.sub(r'[\n\r]', ' ', docstring)
        # eg. change {@link #informListeners(Solution, String)} to informListeners
        docstring = re.sub(r'\{@\w+\s+#?(\w+)\(.*\)\}', r' \g<1> ', docstring)
        # remove website links eg. http://doesciencegrid.org/public/pbs/qsub.html removed
        docstring = re.sub(r'https?://.*?(\s|"|>)', ' ', docstring)
        # eg. change com.openv.spring.SBusinessExample16Remote.getStr(String args) to getStr
        docstring = re.sub(r'(\w+\.){2,}(\w+)\s*\(.*?\)', r' \g<2> ', docstring)
        # eg. remove <code> </code> <p>
        docstring = re.sub(r'<.*?>', '', docstring)
        # eg. remove @param、@return、@throws、@see
        docstring = re.sub(r'@\w+', '', docstring)
        # remove [] bracket and content in it eg. [Upload/Download]
        docstring = re.sub(r'\[.*?\]', '', docstring)
        # remove .jar .zip &quot;
        docstring = re.sub(r'\.\w+|&quot;', '', docstring)
        # remove version number eg. 1_2_2 or 3.0
        docstring = re.sub(r'((\d+_)+\d)|((\d+\.)+\d)', '', docstring)
        # eg. remove => , in the line end /
        docstring = re.sub(r'=>|(/$)', '', docstring)

        # retain only alphabet, digits, and, or, not, + - * / % = < > ?, this symbol may have meanings
        # and "-" may be prefix symbol or just connect between alphabet
        docstring = re.sub(r'[^A-Za-z0-9&|!+\-*/%=<>?]', ' ', docstring)
        # remove 2 or more continuous space
        docstring = re.sub(r' {2,}', ' ', docstring)
        # split word with java camel naming convention
        docstring = re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', docstring)
        docstring = docstring.lower()
        return docstring
        pass

    @classmethod
    def filter_func_name(cls, func_name):
        # remove line delimiter
        func_name = re.sub(r'[\n\r]', ' ', func_name)
        # retain only alphabet, digits, and, or, not, + - * / % = < > ?, this symbol may have meanings
        # and "-" may be prefix symbol or just connect between alphabet
        func_name = re.sub(r'[^A-Za-z0-9]', ' ', func_name)
        # remove 2 or more continuous space
        func_name = re.sub(r' {2,}', ' ', func_name)
        # split word with java camel naming convention
        func_name = re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', func_name)
        func_name = func_name.lower()
        return func_name

    @classmethod
    def __get_wordnet_pos(cls, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return cls.__tag_dict.get(tag, wordnet.NOUN)

    @classmethod
    def tokenize_stop_words_removal_and_lemmatization(cls, code):
        word_list = nltk.word_tokenize(code)
        return " ".join([cls.__lemmatizer.lemmatize(w, cls.__get_wordnet_pos(w)) for w in word_list if
                         (w not in string.punctuation) and (w not in cls.__stop_words)])
