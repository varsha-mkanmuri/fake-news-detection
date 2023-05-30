from xgboost import XGBClassifier
import time
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

import numpy as np
import os
import json

import transformers

import LLM_embeddings as LLM_embeddings

'''
RUNNING THIS MODULE AS PART OF THE PIPELINE WILL TAKE A VERY LONG TIME.
'''

def train_xgboost(X, y):

    xgb_params_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64],
        'n_estimators': [50, 100, 150, 200, 250, 300, 500, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
        'reg_lambda': [0.01, 0.05, 0.1, 0.5, 1.0]
    }

    xgb_search = GridSearchCV(XGBClassifier(), xgb_params_grid, cv=10, \
                              verbose=1, scoring='f1_weighted')

    start = time.time()
    xgb_search.fit(X, y)
    end = time.time()

    print("total time elapsed for 1 set of hyper-parameter eval in XGBoost: {} seconds".format(end - start))

    print("XGBoost optimal_hyperparameters: ", xgb_search.best_params_)
    print("XGBoost best_cv_f1_score: ", xgb_search.best_score_)

    xgb_final = xgb_search.best_estimator_
    xgb_final.fit(X, y)

    file_path = 'trained_models'
    joblib.dump(xgb_final, os.path.join(file_path, 'xgb_model_new.pkl'))

    return xgb_final


def run_llm_xgboost_pipeline(shuffled_df):

    data_X = shuffled_df['text'].copy()
    data_y = shuffled_df['label'].copy()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)

    # If needed - Use any other LLM like  roberta-large / bert-base-uncased etc.
    roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base-openai-detector')

    X_train_encoded = LLM_embeddings.gen_LLM_embeddings(X_train, roberta_tokenizer)
    X_test_encoded = LLM_embeddings.gen_LLM_embeddings(X_test, roberta_tokenizer)

    print(" RoBERTa + XGBoost results ")

    xgb_final = train_xgboost(X_train_encoded, y_train)

    xgb_final_y_train_pred = xgb_final.predict(X_train_encoded)
    xgb_final_y_test_pred = xgb_final.predict(X_test_encoded)

    print("accuracy train: ", accuracy_score(y_train, xgb_final_y_train_pred))
    print("accuracy test: ", accuracy_score(y_test, xgb_final_y_test_pred))

    print("f1score train: ", f1_score(y_train, xgb_final_y_train_pred))
    print("f1score test: ", f1_score(y_test, xgb_final_y_test_pred))

    print(classification_report(y_test, xgb_final_y_test_pred, labels=[0, 1]))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, xgb_final_y_test_pred))

    return xgb_final
