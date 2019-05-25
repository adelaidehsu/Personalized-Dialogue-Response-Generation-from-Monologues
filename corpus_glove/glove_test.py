import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove


sentences = list(itertools.islice(Text8Corpus('./data/op+fri.txt'),None))
corpus = Corpus()
corpus.fit(sentences, window=30)
glove = Glove.load('corpus_op+fri.model')
glove.add_dictionary(corpus.dictionary)
print(glove.most_similar('monica', number=10))
print(corpus.dictionary['monica'])
