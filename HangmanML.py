# This script creates, compiles, and trains the model
# based on the data set 'words_250000_train.txt'.
# At the end it will ask the user if they want to save the model
# as 'hangman_model.keras'.

# As of 2/23/24, this model is still a work in progress. I am currently
# working on adjusting the training to incorporate when some letters are
# revealed and some aren't in a random fashion, such that it learns
# patterns based on what letters are revealed and their location.

import numpy as np
import pandas as pd
import re
# Check if pandas works (it does)
#print(pd.__version__)
import matplotlib as plt
# Find the directory where matplotlib is being imported from
#print(plt.__file__)
# Check if matplotlib works (it does)
#print(plt.__version__)
import tensorflow as tf
# Check if tensorflow works (it does)
#print(tk.__version__)
import random



# Import the training words to a list (word_list):
word_list = []
words_file = open('words_250000_train.txt', 'r')
data = words_file.read()
word_list = data.split('\n')
words_file.close()
# Determine the length of the longest word
# for input data structure of model
n = max(len(word) for word in word_list)
# Make sure n definition worked properly (it did):
#print(n)
# Check if data import worked (it did):
#print(word_list[2])
# Pad all the shorter words with an underscore '_':
padded_data =[word.ljust(n, '_') for word in word_list]
# Check if words got properly padded (it did):
'''print('Length of max word = ', n)
print('Randomly selected word = ', word_list[2])
print('Randomly selected padded word = ', padded_data[2])
print('Length of randomly selected word = ', len(padded_data[2]))'''

# Partition the word_list into training and test sets:
# First randomize the word list,
# new training set should not alter performance.
# Training set is 75% of words.
random.shuffle(padded_data)
word_list_train = padded_data[0:int(len(padded_data)*0.75)]
word_list_test = padded_data[int(len(padded_data)*0.75)::]

# Check if the above worked (it did):
#print(len(word_list_train))
#print(len(word_list_test))

# Convert the training and test sets into obscured words
# to feed to the tensorflow model. The obscured letters
# are represented by a '.'. The padded parts of the letters
# should not be obscured, so only lowercase alphanumeric
# characters are replaced.
def replace_lower_alpha_with_period(string):
    return re.sub(r'[a-z0-9]', '.', string)
replace_vectorized = np.vectorize(replace_lower_alpha_with_period)
X_train_obscure = replace_vectorized(word_list_train)
X_test_obscure = replace_vectorized(word_list_test)
# Check if the above worked (it did):
#print(word_list_train[2])
#print(X_train_obscure[2])
#print(word_list_test[2])
#print(X_test_obscure[2])

# Create the one-hot vectors representing the true values
# for training the model (y_train). Note that we must do this
# for both the training set and the test (validation) set:
# Define the alphabet and padding character
alphabet = 'abcdefghijklmnopqrstuvwxyz'
padding_char = '_'
# Create one-hot encoded vectors for each character in each word
# of the training set
one_hot_vectors = []
for word in word_list_train:
    word_one_hot = []
    for char in word:
        if char in alphabet:
            one_hot = [int(char == letter) for letter in alphabet]
            #^ One-hot encode alphabetic characters
        else:
            one_hot = [0] * len(alphabet)
            #^ Zero vector for padding character
        one_hot.append(int(char == padding_char))
        #^ Add padding character indicator
        word_one_hot.append(one_hot)
    # Pad with zero vectors if necessary to match max_length
    word_one_hot += [[0] * len(one_hot) for _ in range(n - len(word))]
    one_hot_vectors.append(word_one_hot)
# Convert the list of lists to a numpy array
y_train = np.array(one_hot_vectors)
# Reshape the array to match the desired shape (L, n, 27)
y_train = y_train.reshape(len(word_list_train), n, len(alphabet) + 1)
# +1 for the padding character indicator
# Test if the above worked (it did):
#print(word_list_train[-1], ': the word to be one hot vectored')
#print('The one hot vectors of the word:', y_train[-1])

# Now do it again for the test set
one_hot_vectors = []
for word in word_list_test:
    word_one_hot = []
    for char in word:
        if char in alphabet:
            one_hot = [int(char == letter) for letter in alphabet]
            #^ One-hot encode alphabetic characters
        else:
            one_hot = [0] * len(alphabet)
            #^ Zero vector for padding character
        one_hot.append(int(char == padding_char))
        #^ Add padding character indicator
        word_one_hot.append(one_hot)
    # Pad with zero vectors if necessary to match max_length
    word_one_hot += [[0] * len(one_hot) for _ in range(n - len(word))]
    one_hot_vectors.append(word_one_hot)
# Convert the list of lists to a numpy array
y_test = np.array(one_hot_vectors)
# Reshape the array to match the desired shape (L, n, 27)
y_test = y_test.reshape(len(word_list_test), n, len(alphabet) + 1)
# Test if the above worked (it did):
#print(word_list_test[-1], ': the word to be one hot vectored')
#print('The one hot vectors of the word:', y_test[-1])


# Create the model
#
# TO-DO LIST:
# [x] Create the mask for the padded words
# [x] Create the one hot 'y_train' vectors
# [x] Define the input shape parameter n where n = length of longest word
# [x] Define the input shape (n x 28 for n-letter obscured word)
# 28 for the 26 characters plus the obscuring character '.' and the
# padding character '_'
# [x] Convert the obscured X_train set to the input shape
# [x] Make the training loop for the model to play the game
# [] Error hunting
# [] Model optimization
#input_shape = (None, n, 2)
input_shape = (n, 2)

# Convert the obscured X_train and X_test to 2D one hot vectors
# where index 0 = '.' and index 1 = '_'
one_hot_train = []
for word in X_train_obscure:
    one_hot_train_word = []
    for char in word:
        if char == '.':
            one_hot_train_word.append([1, 0])
        if char == '_':
            one_hot_train_word.append([0, 1])
    one_hot_train.append(one_hot_train_word)
X_train_obscure = np.array(one_hot_train)
one_hot_test = []
for word in X_test_obscure:
    one_hot_test_word = []
    for char in word:
        if char == '.':
            one_hot_test_word.append([1, 0])
        if char == '_':
            one_hot_test_word.append([0, 1])
    one_hot_test.append(one_hot_test_word)
X_test_obscure = np.array(one_hot_test)

# Mask the underscores of the padded words
mask = np.zeros_like(X_train_obscure)
mask[:, :, -1] = 1 # Assuming the last dimension represents padded character '_'

print('Made it to initialization of the model...')

# Define the model architecture
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
# ^Start the weights of the node connections with a normal distribution
# around 0 with stdev of 0.01
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Flatten(),  # Flatten the input tensor
    tf.keras.layers.Masking(mask_value=1), # Masking padded values
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer), # Output layer with flattened output
    tf.keras.layers.Dense(29 * 27, activation='softmax', kernel_initializer=initializer), # Reshape the output to match the desired shape
    tf.keras.layers.Reshape((29, 27))
])

print('Model initialized, compiling now...')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Model is about to begin training...')

# Train the model:
model.fit(X_train_obscure, y_train, epochs=10, batch_size=32, verbose=2)    

print('Model is being evaluated...')

# Evaluate the model
loss, accuracy = model.evaluate(X_test_obscure, y_test, verbose=0)
print("Test Accuracy:", accuracy)

#Save the model if the user wishes to
ans = input('Do you want to save this model? Please note that if you do it will overwrite the previous model (Y/N): ')
if ans == 'Y':
    model.save('hangman_model.keras')


