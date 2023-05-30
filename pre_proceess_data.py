import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# Code to pre-process text data

def lowercase(text):
    """
    Convert text to lowercase.
    """
    return text.lower()

def tokenize(text):
    """
    Tokenize text into individual words.
    """
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    """
    Remove stopwords from list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def stem(tokens):
    """
    Stem words in list of tokens.
    """
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def lemmatize(tokens):
    """
    Lemmatize words in list of tokens.
    """
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lem_tokens

def remove_special_chars(text):
    """
    Remove special characters and punctuation marks from text.
    """
    cleaned_text = re.sub('[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def remove_urls(text):
    """
    Remove URLs from text.
    """
    cleaned_text = re.sub(r'http\S+', '', text)
    return cleaned_text


def pre_proccess(text):
    text = lowercase(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    tokens = lemmatize(tokens)
    text = ' '.join(tokens)
    text = remove_special_chars(text)
    text = remove_urls(text)
    return text


def preprocess_pipe(shuffled_df):
    print("start pre-processing text")
    shuffled_df['text'] = shuffled_df['text'].progress_apply(pre_proccess)
    print("pre-processing done")

    return shuffled_df