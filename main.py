import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import seaborn as sns

import read_data as read_data
import word2vec_svm_pipeline as word2vec_svm_pipeline
import llm_lstm_nn_pipeline as llm_lstm_nn_pipeline
import llm_xgboost_pipeline as llm_xgboost_pipeline

df_true, df_fake, shuffled_df = read_data.get_isot_dataset()

#  Including without hyper-parameter tuning code
# Training the below algorithms will take a very long time - with hyper-parameter tuning.
# Refer tuning scripts, for experimentation with hyper-parameter tuning

print("Starting Word2Vec SVM pipeline process")
rbfsvm = word2vec_svm_pipeline.run_word2vec_svm(shuffled_df)
print("Finished Running Word2Vec SVM pipeline")

# Hyper-parameter tuning part not included as part of NN pipeline, as it will take several days to run.
# Experiment with different RNN, LSTM, CNN with different layers / units etc.
print("Starting RoBERTa + LSTM Neural Networks pipeline process")
nn_model = llm_lstm_nn_pipeline.run_lstm_nn_pipeline(shuffled_df)
print("Finished Running RoBERTa + LSTM Neural Networks pipeline")

# Hyper-parameter tuning part not included as part of the XGBoost pipeline, as it will take several days to run.
# Tuning number of estimators, lambda, gamma, max_depth etc.
print("Starting RoBERTA + XGBoost pipeline process")
xgb_final = llm_xgboost_pipeline.run_llm_xgboost_pipeline(shuffled_df)
print("Finished Running RoBERTA + XGBoost pipeline")



