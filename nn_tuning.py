from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.models import Model, Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import json

import transformers

import LLM_embeddings as LLM_embeddings

'''
Params grid to tune the neural network architecture.

Running the llm_lstm_nn_pipeline - while tuning the below hyper-parameters will take a long time
'''

nn_params_grid = {
    'units': [64, 128, 256, 512, 1024],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'batch_size': [4, 16, 32, 64, 128, 256, 512],
    'epochs': [50, 100, 150],
    'hidden_layer_sizes': [(500,), (1024, 512), (512, 512), (512, 256), (2048, 1024, 512)],
    "solver": ['adam', 'sgd'],
    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
}

# Training for more than 10 epochs or smaller learning rate, or larger NN architectures -  will take a very long time.

def train_lstm_nn_arch(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1):
    model = Sequential()

    # Add the first hidden layer
    model.add(LSTM(units=256))
    model.add(Dense(256, activation='relu'))
    # Add the second hidden layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_reshape = X.reshape(X.shape + (1,))
    X_reshape = X_reshape.astype('float32')

    # Train the model

    history = model.fit(X_reshape, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join("trained_models", "nn_model", "nn_model.json"), "w") as json_file:
        json_file.write(model_json)

    json_file.close()
    # serialize weights to HDF5
    model.save_weights(os.path.join("trained_models", "nn_model", "nn_model.h5"))
    print("")
    print("Saved LSTM NN model to \" trained models \\ nn_model \" directory")
    print("")

    return model

def run_lstm_nn_pipeline(shuffled_df):

    data_X = shuffled_df['text'].copy()
    data_y = shuffled_df['label'].copy()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42, stratify=data_y)

    # If needed - Use any other LLM like  roberta-large / bert-base-uncased etc.
    roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base-openai-detector')

    X_train_encoded = LLM_embeddings.gen_LLM_embeddings(X_train, roberta_tokenizer)
    X_test_encoded = LLM_embeddings.gen_LLM_embeddings(X_test, roberta_tokenizer)

    print(" RoBERTa + LSTM NN results ")

    nn_model = train_lstm_nn_arch(X_train_encoded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model on the test data
    loss, accuracy = nn_model.evaluate(X_test_encoded, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_nn_pred_vals = nn_model.predict(X_test_encoded)
    y_out = np.round(y_nn_pred_vals)

    print(classification_report(y_test, y_out, labels=[0, 1]))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_out))

    return nn_model
