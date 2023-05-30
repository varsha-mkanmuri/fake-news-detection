from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

import word2vec_embeddings as word2vec_embeddings
import ml_svm as ml_svm
import joblib
import os

'''
We do not need tune this model further - as it already has great train/test performance nearly ~100%.
'''

def run_word2vec_svm(shuffled_df):
    data_X = shuffled_df['text'].copy()
    data_y = shuffled_df['label'].copy()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)

    word2vec_model = word2vec_embeddings.train_word2vec_emb(X_train, dim_len=300)

    X_train_word = word2vec_embeddings.get_word2vec_emb(X_train, word2vec_model, dim_len=300)
    X_test_word = word2vec_embeddings.get_word2vec_emb(X_test, word2vec_model,  dim_len=300)

    rbfsvm = ml_svm.svm_train(X_train_word, y_train)

    rbfsvm_y_train_pred = ml_svm.svm_predict(X_train_word, rbfsvm)
    rbfsvm_y_test_pred = ml_svm.svm_predict(X_test_word, rbfsvm)

    print("RBF_SVM + Word2Vec results ")
    print(classification_report(y_test, rbfsvm_y_test_pred, labels=[0, 1]))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, rbfsvm_y_test_pred))

    print("accuracy train: ", accuracy_score(y_train, rbfsvm_y_train_pred))
    print("accuracy test: ", accuracy_score(y_test, rbfsvm_y_test_pred))

    print("f1score train: ", f1_score(y_train, rbfsvm_y_train_pred))
    print("f1score test: ", f1_score(y_test, rbfsvm_y_test_pred))

    # Save the model to a file
    file_path = 'trained_models'
    joblib.dump(rbfsvm, os.path.join(file_path, 'rbfsvm_model.pkl'))

    return rbfsvm

