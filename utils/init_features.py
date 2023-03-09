from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

def nltk_tokenizer(code_snippet):
    return nltk.word_tokenize(code_snippet)

def load_wv(wv_path):
    return Word2Vec.load(wv_path).wv