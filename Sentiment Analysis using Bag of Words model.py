# required libraries

from nltk.corpus import stopwords
from collections import Counter
from os import listdir
import numpy as np
import string
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense


#-------------------------------------------------------------------#
#                   DATA PREPROCESSING                              # 
#-------------------------------------------------------------------#

# training and test data paths
train_set_pos_path = "Data/Training Data/pos/"
train_set_neg_path = "Data/Training Data/neg/"

test_set_pos_path = "Data/Test Data/pos/"
test_set_neg_path = "Data/Test Data/neg/"

# to load the contents of the file 
def load_file(filepath):
    file = open(filepath, 'r')  # open the file in the read only mode
    text = file.read()          # read the contents of the file
    file.close()                # close the file 
    return text

# to clean the file for tokens
def clean_file(file):
    tokens = file.split()                                 # split into tokens on whitespace
    
    table = str.maketrans('' , '', string.punctuation)    # remove punctuation
    tokens = [w.translate(table) for w in tokens]
    
    tokens = [word for word in tokens if word.isalpha()]  # remove non-alphabetic tokens
    
    set_of_stop_words = set(stopwords.words('english'))   # remove stop words
    tokens = [word for word in tokens if not word in set_of_stop_words]
    
    tokens = [word for word in tokens if len(word) > 1]   # remove tokens of length <= 1
    
    return tokens

# to define a vocabulary of words
def add_words_to_vocab_and_update_count(directory, vocab):
    for filename in listdir(directory):
        filepath = directory + '/' + filename
        text = load_file(filepath)  # load the file
        tokens = clean_file(text)   # clean the file
        vocab.update(tokens)        # update count of the word in the vocab
    
vocab = Counter()   # to hold tokens and their respective counts. Eg: [('tok1',tok1_count), ('tok2',tok2_count),...]

add_words_to_vocab_and_update_count('Data/Training Data/pos', vocab)
add_words_to_vocab_and_update_count('Data/Training Data/neg', vocab)

print('The length of the vocab: ',len(vocab))
print('\nTop 10 frequently occuring words:',vocab.most_common(10))

min_occurrence = 2

print('Total Tokens (Before): ',len(vocab))
tokens = [token for token,count in vocab.items() if count >= min_occurrence]    # list of tokens with count >= 2
print('Total Tokens (After) : ',len(tokens))

# to write to the file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

save_list(tokens, 'vocab.txt')   # the vocabulary is saved in a text file for later use

#-------------------------------------------------------------------#
#                   BAG OF WORDS REPRESENTATION                     # 
#-------------------------------------------------------------------#

# to generate list of reviews
def reviews_to_lines(directory, vocab):
    lines = []
    for filename in listdir(directory):
        filepath = directory + filename
        text = load_file(filepath)  # load the file
        tokens = clean_file(text)   # clean the file
        tokens = [word for word in tokens if word in vocab]   # filter by vocab
        line = ' '.join(tokens)     # single review -> tokens -> filter -> single line with tokens spaced by whitespace
        lines.append(line)          # list of reviews. Single review is stored at each index of the list
    return lines

# load the vocabulary
vocab = load_file("vocab.txt")
vocab = vocab.split()
vocab = set(vocab)

# Training Data : reviews to lines
train_pos_reviews_to_lines = reviews_to_lines(train_set_pos_path, vocab)
train_neg_reviews_to_lines = reviews_to_lines(train_set_neg_path, vocab)

# Test Data : reviews to lines
test_pos_reviews_to_lines = reviews_to_lines(test_set_pos_path, vocab)
test_neg_reviews_to_lines = reviews_to_lines(test_set_neg_path, vocab)

# Total training and testing data
train_reviews = train_pos_reviews_to_lines + train_neg_reviews_to_lines
test_reviews  = test_pos_reviews_to_lines  + test_neg_reviews_to_lines

# to prepare the data for training using Bag-Of-Words model
def prepare_data(train_reviews, test_reviews, mode):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_reviews) # fit the tokenizer on the texts

    xtrain = tokenizer.texts_to_matrix(train_reviews, mode = mode)  # encode the training set
    xtest  = tokenizer.texts_to_matrix(test_reviews, mode = mode)   # encode the test set

    return xtrain, xtest

xtrain, xtest = prepare_data(train_reviews, test_reviews, mode = 'freq')

print(" Shape of xtrain: ", xtrain.shape)
print(" Shape of xtest : ", xtest.shape)

train_pos_limit = int(xtrain.shape[0]/2) # upper limit of pos training reviews
train_neg_limit = xtrain.shape[0]        # upper limit of neg training reviews
test_pos_limit  = int(xtest.shape[0]/2)  # upper limit of pos test reviews
test_neg_limit  = xtest.shape[0]         # upper limit of neg test reviews 

ytrain = np.array([0 for i in range(0, train_pos_limit)] + [1 for i in range(train_pos_limit, train_neg_limit)])
ytest  = np.array([0 for i in range(0, test_pos_limit)]  + [1 for i in range(test_pos_limit, test_neg_limit)])

#-------------------------------------------------------------------#
#                   SENTIMENT ANALYSIS MODEL                        # 
#-------------------------------------------------------------------#

# the training/learning model
def seniment_analysis_model(xtrain, ytrain):
    n_words = xtrain.shape[1]  
    # define the network
    model = Sequential()
    model.add(Dense(50, input_shape = (n_words, ), activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    # compile the network
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # fit the network to the training data
    history = model.fit(xtrain, ytrain, epochs = 25, verbose = 2)
    
    return model, history

classifier, model_history = seniment_analysis_model(xtrain, ytrain)

# evaluation of the preformance of the trained model on the test set
loss, accuracy = classifier.evaluate(xtest, ytest, verbose = 0)
print('Test accuracy = ', (accuracy * 100))
