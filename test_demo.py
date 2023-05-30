import joblib
import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import time
import json

from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tqdm
from gensim.models import Word2Vec


import word2vec_embeddings as word2vec_embeddings
import LLM_embeddings as LLM_embeddings

file_path = 'trained_models'


# Load the models

# Load Word2vec model
word2vec_model = Word2Vec.load(os.path.join(file_path, 'word2vec_model', 'word2vec_model.model'))

# If needed - Use any other LLM like  roberta-large / bert-base-uncased etc.
roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base-openai-detector')

# SVM Word2Vec
rbfsvm = joblib.load(os.path.join(file_path, 'rbfsvm_model.pkl'))


# RoBERTa + Neural Network
json_file = open(os.path.join("trained_models", "nn_model", "nn_model.json"), "r")
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)
# load weights into new model
nn_model.load_weights(os.path.join("trained_models", "nn_model", "nn_model.h5"))
print("Loaded LSTM NN model from disk")
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# RoBERTa + XGBoost
xgb_final = joblib.load(os.path.join(file_path, 'xgb_model_new.pkl'))


# Input Sample -  Can be given from Command line prompt through terminal
# Example (For True - Real data ):-   python test_demo.py sample_true1.csv
# Example (For Fake - synthetic data ):-   python test_demo.py sample_fake_syn_1.csv
# Change contents in the '.csv' files - to add new/other types of input news.
# Give only 1 example as input at a time, Else use the other ML pipleline to evaluate metrics

df_sample = pd.read_csv(os.path.join('sample_demo', sys.argv[1]))

print("Running the different NLP models + ML/DL classifiers for example in \"text\" column in file - ", sys.argv[1])
print("")



X_syn_word = word2vec_embeddings.get_word2vec_emb(df_sample['text'], word2vec_model, dim_len=300)

X_syn_LLM = LLM_embeddings.gen_LLM_embeddings(df_sample['text'], roberta_tokenizer)


dict_label = {1: "fake news", 0: "real news"}

y_output_rbfsvm_word2vec = rbfsvm.predict(X_syn_word)
print("Test Demo output - Prediction of rbf_svm + word2vec model: ", dict_label[y_output_rbfsvm_word2vec[0]])


y_output_roberta_neural_networks = nn_model.predict(X_syn_LLM)
y_nn_out = np.round(y_output_roberta_neural_networks)[0]
print("Test Demo output - Prediction of RoBERTa + LSTM Neural Networks model: ", dict_label[int(y_nn_out[0])])

y_output_roberta_xgboost = xgb_final.predict(X_syn_LLM)
print("Test Demo output - Prediction of RoBERTa + XGBoost model: ", dict_label[y_output_roberta_xgboost[0]])

