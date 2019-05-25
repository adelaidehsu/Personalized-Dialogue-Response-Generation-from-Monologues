import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove

sentences = list(itertools.islice(Text8Corpus('./data/op+fri.txt'),None))
corpus = Corpus()
corpus.fit(sentences, window=30)
glove = Glove(no_components=384, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
print(glove.most_similar('man', number=10))
glove.save('corpus_op+fri_512.model')
