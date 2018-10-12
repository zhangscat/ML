# binary classification example
# classifying movie reviews as positive or negative
# equal number of reviews from training and test mean data is balanced
# uses keras--high level API--to train models in tensor flow

import tensorflow as tf
from tensorflow import keras

import numpy as numpy
print(tf.__version__)

# IMDB dataset has already been preprocessed such that reviews have been converted
# to sequences of integers where each integer represents a specific word in a dict
# Download IMDB dataset
# STEP 1: GATHER DATA

imdb = keras.datasets.imdb # what other keras datasets are there?
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

# mum_words = 10,000 keeps 10,000 most frequently occurring words in the training data
# STEP 2: EXPLORE YOUR DATA
# Each example is an array of ints representing the words of the movie review
# Each label is either 0 (negative review) or (postive review) 1
# num of training and num of testing reviews
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# Display the first review
print(train_data[0])
# Display length of first two movie reviews
print(len(train_data[0]), test_data[1])
# Since inputs to neural network must be same length, review length must be fixed

# CONVERT INTS BACK TO WORDS (OPTIONAL?)
# Create a helper func to convert ints back to text
# A dictionary mapping words to an int index
word_index = imdb.get_word_index()

# First indices are reversed
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Use decode_review to display text for first review
decode_review(train_data[0])

# STEP 2.5: CHOOSE A MODEL

# STEP 3: PREPARE THE DATA
