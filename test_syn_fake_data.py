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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

import word2vec_embeddings as word2vec_embeddings
import LLM_embeddings as LLM_embeddings

file_path_data = 'syn_fake_data'

df1 = pd.read_csv(os.path.join(file_path_data, 'fake_syn_data.csv'))
df2 = pd.read_csv(os.path.join(file_path_data, 'fake_syn_data_new.csv'))
df3 = pd.read_csv(os.path.join(file_path_data, 'fake_syn_data_new_2.csv'))

df_fake_final = pd.concat([df1,df2,df3])
df_fake_final = df_fake_final.reset_index(drop=True)

# Final fake data generated is saved here.
df_fake_final.to_csv(os.path.join(file_path_data,'fake_final.csv'), index=False)


# Replace with the 500 generated file - if re-generating new data - or any file you want.
# Include the news in a 'text' column of the csv file and it will work.
df_fake_final = pd.read_csv(os.path.join(file_path_data, 'fake_final.csv'))


# Load the models
file_path = 'trained_models'

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



print("Running the different NLP models + ML/DL classifiers for the generated 500 synthetic fake new data")
print("")



X_syn_word = word2vec_embeddings.get_word2vec_emb(df_fake_final['text'], word2vec_model, dim_len=300)

X_syn_LLM = LLM_embeddings.gen_LLM_embeddings(df_fake_final['text'], roberta_tokenizer)


y_ground_fake_vals = df_fake_final['label']


rbfsvm_y_fake_pred = rbfsvm.predict(X_syn_word)
print("RBF SVM + Word2Vec accuracy score on fake synthetic data: ", accuracy_score(y_ground_fake_vals, rbfsvm_y_fake_pred))


y_nn_pred_vals = nn_model.predict(X_syn_LLM)
y_out = np.round(y_nn_pred_vals)
print("RoBERTa + LSTM Neural Networks accuracy score on fake synthetic data: ", accuracy_score(y_ground_fake_vals, y_out))

y_xgb_pred_vals = xgb_final.predict(X_syn_LLM)
print("RoBERTa + XGBoost accuracy score on fake synthetic data: ", accuracy_score(y_ground_fake_vals, y_xgb_pred_vals))