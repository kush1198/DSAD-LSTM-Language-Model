from __future__ import print_function
import numpy as np
from gensim.models import Word2Vec
import string
from lstmnumpy import lstmnumpy

bptt = 40
embedding_size = 64
print('\nFetching the text...')
with open('arxiv_abstracts.txt') as f:   # Reading the File
    docs = f.readlines()    # List of all lines in file ,making total of 7200 lines.
    print('\nPreparing the sentences...')
    sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:bptt]] for doc in docs]
    ## For every sentence , all the words are made in a list of 40, excluding rest.
    ## Shape of sentences (7200,40)

print('Num sentences:', len(sentences))

print('\nTraining word2vec...')
word_model = Word2Vec(sentences, size=embedding_size, min_count=1, window=5, iter=100)
    #   gensim's word2vec expects sentences as i/p,each having a list of words
    #   min_count= atleast this no of times, word should come
    #   size->Embedding size :Bigger size values require more training data, but can lead to better
    #   models. Reasonable values are in the tens to hundreds.
    #   window (int, optional) – Maximum distance between the current and predicted word within a sentence.

embedding_matrix = word_model.wv.vectors
    #   Returns embedding matrix from trained word_model
print('Result embedding shape:', embedding_matrix.shape)
    #   Of shape (word,Embeddings Value)

def word2idx(word):
  return word_model.wv.vocab[word].index

def idx2word(idx):
  return word_model.wv.index2word[idx]

print('\nPreparing the data for LSTM...')

train_x = np.zeros([len(sentences), embedding_size, bptt])   # 7200,64,40
train_y = np.zeros([len(sentences)])    # 7200

for i, sentence in enumerate(sentences):    # Iterates over Sentences with index of each sentence.
  for t, word in enumerate(sentence[:-1]):  # Iterates over each word with index of each word
      for j in range(0,embedding_size):     
          train_x[i,j,t] = word_model.wv[word][j]   # Initialize each embedding to its respective place in train_x
  train_y[i] = word2idx(sentence[-1]) # For every sentence, word2idx is used to return index of each word in word2vec.
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print(train_x[10,:,:].shape)



# Code that needs to be reformatted after completition of lstmnumpy
trainsample = train_x[10,:,:]


layer1 = lstmnumpy(hidden_size = 64, embedding_size = embedding_size, bptt = bptt, activation = 'sigmoid', wordmodel = word_model)
layer2 = lstmnumpy(hidden_size = 64, embedding_size = embedding_size, bptt = bptt, activation = 'sigmoid', wordmodel = word_model)
layer3 = lstmnumpy(hidden_size = 64, embedding_size = embedding_size, bptt = bptt, activation = 'softmax', last = True, wordmodel = word_model)

A = trainsample
cache = {}
model = [layer1, layer2, layer3]
for layer in model:
    Aprev = A
    A, cache = layer.forward_sequence(Aprev, train_y[10])
    

#cachenew = cache
#for t in reversed(range(len(model))):
    
    
print(A)
    
    
