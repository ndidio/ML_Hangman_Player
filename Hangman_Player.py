# This script creates the Hangman game and then uses the model from
# HangmanML.py to play it. The model is called 'hangman_model.keras'.

# As of 2/23/24, the model is still a work in progress. I am currently
# working on adjusting the training to incorporate when some letters are
# revealed and some aren't in a random fashion, such that it learns
# patterns based on what letters are revealed and their location.

# Define n, the length of the longest word the data was trained on.
# This is to format the input for the model
n = 29
# This is to convert the user word to an obscured one
from tensorflow.keras.models import load_model
import numpy as np
import re
def replace_lower_alpha_with_period(string):
    return re.sub(r'[a-z0-9]', '.', string)
import time

# Create the actual hangman game, a wrapper for the model
# that handles the dynamics of gameplay

# Now below we are actually going to play the game using the
# already trained model saved as 'hangman_model.keras'
model = load_model('hangman_model.keras')
user_word = input("Please enter a word to play. Only lowercase please: ")
len_word = len(user_word)
# Obscure and pad the user word so it can be injected into the model
user_obscured = replace_lower_alpha_with_period(user_word)
user_obscured_display = list(user_obscured) # For display purposes
output_scoreboard = user_obscured # For display purposes
# Obscuring
user_obscured = re.sub(r'[a-z0-9]', '.', user_word)
# Padding
user_obscured = user_obscured.ljust(n, '_')
user_word_input = user_word
user_word = user_word.ljust(n, '_')
# One-hot the user word as 'encoded'
alphabet = 'abcdefghijklmnopqrstuvwxyz'
encoded = np.zeros((n, 27), dtype=np.int32)
i = 0
for char in user_word:
    if char in alphabet:
        index = ord(char) - ord('a')
        encoded[i, index] = 1
    else:
        encoded[i, -1] = 1
    i += 1
# Check if the above worked (it did):
#print(encoded)

one_hot_obscured = []
# One-hot the obscured user word to make it a viable input
for char in user_obscured:
    if char == '.':
            one_hot_obscured.append([1, 0])
    if char == '_':
            one_hot_obscured.append([0, 1])
# Remember to convert all inputs to the model to numpy arrays
one_hot_obscured = np.array(one_hot_obscured, dtype = int)
# Fix and check the shape of one_hot_obscured (it is correct):
one_hot_obscured = one_hot_obscured.reshape(1, 29, 2)
#print(one_hot_obscured)
#print(one_hot_obscured.shape)

lives = 6
correct_response_num = 0
previous_guesses = np.zeros(27)
num_guesses = 0
print('Starting the game now...')
time.sleep(0.5)
while correct_response_num < len_word and lives > 0:
    num_guesses += 1
    guess = model.predict(one_hot_obscured, verbose = 2)
    # The below few lines finds the character that the model wishes
    # to guess, since the model outputs guesses based on position as
    # well.
    #print(guess.shape)
    index_large = np.unravel_index(np.argmax(guess), guess.shape)
    index_position = index_large[1]
    index_large = index_large[2]
    # The following loop checks if a guess has been made
    # if it has, it defaults to the next most probable argument
    # of the model, and continues until it hits a character that
    # has yet to be guessed. It also stops the model from guessing
    # the padded character '_'.
    while previous_guesses[index_large] == 1 or index_large == 26:
        index_to_zero = index_large
        guess[0, index_position, index_to_zero] = 0
        index_large = np.argmax(guess)
        index_large = np.unravel_index(np.argmax(guess), guess.shape)
        index_position = index_large[1]
        index_large = index_large[2]
    # Now the letter to guess is guess[index_large] and it has
    # not been guessed before.
    previous_guesses[index_large] = 1
    # Now turn the guess into a one hot vector, then
    # compare it to the encoded user word
    one_hot_guess = [0]*27
    for i in range(len(guess[0, index_position])):
        if i == index_large:
            one_hot_guess[i] = 1
    # Check if the form of one_hot_guess is correct (it is):
    #print(one_hot_guess)
    #print(guess[0, index_position])
    # See what the machine is guessing
    alpha_index = np.argmax(one_hot_guess)
    print('The machine guessed the letter ', alphabet[alpha_index])
    time.sleep(1.5)
    # And now compare it
    if alphabet[alpha_index] in user_word_input:
        print('Correct guess!')
        correct_response_num += 1
        indices = [i for i, char in enumerate(user_word_input) if char == alphabet[alpha_index]]
        for element in indices:
            user_obscured_display[element] = alphabet[alpha_index]
        output_scoreboard = ''.join(user_obscured_display)
    if alphabet[alpha_index] not in user_word_input:
        print('Incorrect guess!')
        lives -= 1
    time.sleep(1.5)
    # Reveal the word as the machine is guessing
    print('The scoreboard so far reads...')
    time.sleep(1.5)
    print(output_scoreboard)
    print('Lives remaining: ', lives)
if correct_response_num == len_word:
    print('The machine won in ', num_guesses, ' guesses!')
if lives == 0:
    print('You won after ', num_guesses, ' guesses by the machine!')
