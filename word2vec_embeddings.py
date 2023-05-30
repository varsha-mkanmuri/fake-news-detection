import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec

# Encode the text using the trained Word2Vec model
def word2vec_encoding(text, word2vec_model):
    features = []
    for word in text:
        if(word in word2vec_model.wv):
            features.append(word2vec_model.wv[word])
    encoding = np.mean(features, axis=0)
    return encoding

def train_word2vec_emb(X, dim_len=300):
    # Tokenize the text using NLTK tokenizer
    X_tokenized = X.apply(nltk.word_tokenize)
    # Train the Word2Vec model
    word2vec_model = Word2Vec(X_tokenized, vector_size=dim_len, min_count=1)

    file_path = 'trained_models'
    word2vec_model.save(os.path.join(file_path, 'word2vec_model', 'word2vec_model.model'))

    return word2vec_model

def get_word2vec_emb(X, word2vec_model, dim_len=300):
    # Tokenize the text using NLTK tokenizer
    X_tokenized = X.apply(nltk.word_tokenize)

    # Encode the data to get word embeddings
    X_encoded = X_tokenized.apply(lambda x: word2vec_encoding(x, word2vec_model))

    cols = ['features_' + str(i) for i in range(dim_len)]
    X_encoded_word = pd.DataFrame(X_encoded.tolist(), columns=[cols])

    return X_encoded_word


