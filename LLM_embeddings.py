# This file generates word embeddings using LLM.
# You can change which LLM you want by replacing the model name from hugging face.

import tensorflow as tf
import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tqdm


def create_tokens(tokenizer_name, docs):
    features = []

    for doc in tqdm.tqdm(docs, desc = 'converting words to features'):
        tokens = tokenizer_name.tokenize(doc)
        ids = tokenizer_name.convert_tokens_to_ids(tokens)
        features.append(ids)

    return features

def gen_LLM_embeddings(X, tokenizer):

    # If needed - Use any other LLM like  roberta-large / bert-base-uncased etc.
    #roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base-openai-detector')

    roberta_tokenizer = tokenizer

    roberta_features = create_tokens(roberta_tokenizer, X)

    roberta_LLM_embeddings = pad_sequences(roberta_features, maxlen=500)

    return roberta_LLM_embeddings

