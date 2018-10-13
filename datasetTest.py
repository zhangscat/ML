#TensorFLow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import csv

print(tf.__version__)

# Import csv file to python array
with open('symptom_test.csv', 'r') as f: 
    symptoms = list(csv.reader(f,
delimiter=';'))
print(symptoms[:3])

# Convert python array to tensor object

# Convert tensor object into datasets


# Or try doing it at once
# Creates a dataset that reads all of the records from two CSV files, each with
# eight float columns
filenames = ["symptom_test.csv", "diagnosis_test.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
