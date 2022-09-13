import pandas as pd
from gensim.models import Word2Vec
#from glove import Corpus, Glove # problems with mac
import fasttext

class WordEmbeddings(object):

    def __init__(self, corpus, model_emb, size=300):
        self.corpus = corpus
        self.model = model_emb
        self.size = size

    def get_embedding_model(self):
        if self.model == 'w2v':
            print('Training word2vec model')
            return self.train_word2vec()
        elif self.model == 'glove':
            print('Training glove model')
            return self.train_glove()
        elif self.model == 'ft':
            print('Training fast Text model')
            return self.train_fast_text()
        else: # ''
            return ''

    def train_word2vec(self):
        model = Word2Vec(self.corpus, size=self.size, window=5, min_count=1, workers=4)
        words = model.wv.vocab
        model_dict = dict()
        for word in words:
            model_dict[word] = model.wv[word]
        return model_dict


    def train_glove(self):
        '''
        corpus = Corpus()
        corpus.fit(self.corpus, window=10)
        glove = Glove(no_components=self.size, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        model_dict = dict()
        for word in glove.dictionary:
            vector = glove.word_vectors[glove.dictionary[word]]
            model_dict[word] = vector
        return model_dict
        '''
        return ''

    def train_fast_text(self):
        path = 'extras/auxiliar.txt'
        # model = fasttext.train_unsupervised(path, model='skipgram')  # , dim=100)
        model = fasttext.train_unsupervised(path, model='skipgram', minCount=1)
        model_dict = dict()
        words = model.words
        for word in words:
            model_dict[word] = model[word]
        return model_dict








