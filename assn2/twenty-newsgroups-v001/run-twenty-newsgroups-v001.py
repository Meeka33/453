# Text Classification Example with Selected Newsgroups from Twenty Newsgroups

# Author: Thomas W. Miller (2019-03-08)

# Compares text classification performance under random forests
# Six vectorization methods compared:
#     TfidfVectorizer from Scikit Learn
#     CountVectorizer from Scikit Learn
#     HashingVectorizer from Scikit Learn
#     Doc2Vec from gensim (dimension 50)
#     Doc2Vec from gensim (dimension 100)
#     Doc2Vec from gensim (dimension 200)

# See example data and code from 
# https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

# The 20 newsgroups dataset comprises around 18000 newsgroups 
# posts on 20 topics split in two subsets: one for training (or development) 
# and the other one for testing (or for performance evaluation). 
# The split between the train and test set is based upon messages 
# posted before and after a specific date.

###############################################################################
### Note. Install all required packages prior to importing
###############################################################################
import multiprocessing

import re,string
from pprint import pprint

import numpy as np

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import nltk
stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = False

from nltk.stem import PorterStemmer
#Functionality to turn stemming on or off
STEMMING = False  # judgment call, parsed documents more readable if False

MAX_NGRAM_LENGTH = 1  # try 1 for unigrams... 2 for bigrams... and so on
VECTOR_LENGTH = 1000  # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = False
SET_RANDOM = 9999

# subsets of newsgroups may be selected
# SELECT_CATEGORY = 'COMPUTERS'
# SELECT_CATEGORY = 'RECREATION'
# SELECT_CATEGORY = 'SCIENCE'
# SELECT_CATEGORY = 'TALK'
SELECT_CATEGORY = 'ALL'

##############################
### Utility Functions 
##############################
# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']    

# text parsing function for entire document string
def parse_doc(text):
    text = text.lower()
    text = re.sub(r'&(.)+', "", text)  # no & references  
    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation  
    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote 
    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings    
    if text.isdigit(): text = ""  # omit words that are all digits    
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
def parse_words(text): 
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]         
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 

##############################
### Gather Original Data 
##############################
newsgroups_all_train = fetch_20newsgroups(subset='train')
# pprint(list(newsgroups_all_train.target_names))
print('\nnewsgroups_all_train.filenames.shape:', newsgroups_all_train.filenames.shape)
print('\nnewsgroups_all_train.target.shape:', newsgroups_all_train.target.shape) 
print('\nnewsgroups_all_train.target[:10]:', newsgroups_all_train.target[:10])

if SELECT_CATEGORY == 'COMPUTERS':
    categories = ['comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 
        'comp.sys.mac.hardware',
        'comp.windows.x']

if SELECT_CATEGORY == 'RECREATION':
    categories = ['rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball', 
        'rec.sport.hockey']

if SELECT_CATEGORY == 'SCIENCE':
    categories = ['sci.crypt',
        'sci.electronics',
        'sci.med', 
        'sci.space']

if SELECT_CATEGORY == 'TALK':
    categories = ['talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc', 
        'talk.religion.misc']

if SELECT_CATEGORY == 'ALL':
    categories = newsgroups_all_train.target_names

print('\nSelected newsgroups:')
pprint(categories) 
# define set of training documents for the selected categories   
# remove headers, signature blocks, and quotation blocks from documents
newsgroups_train_original = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)

print('\nObject type of newsgroups_train_original.data:', 
	type(newsgroups_train_original.data))    
print('\nNumber of original training documents:',
	len(newsgroups_train_original.data))	           
print('\nFirst item from newsgroups_train.data_original\n', 
	newsgroups_train_original.data[0])

# use generic name for target values
train_target = newsgroups_train_original.target

# define set of test documents for the selected categories   
# remove headers, signature blocks, and quotation blocks from documents
newsgroups_test_original = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),	
                                     categories=categories)
print('\nObject type of newsgroups_test_original.data:', 
	type(newsgroups_train_original.data))    
print('\nNumber of test documents:',
	len(newsgroups_test_original.data))	           
print('\nFirst item from newsgroups_test_original.data\n', 
	newsgroups_train_original.data[0])

# use generic name for target values
test_target = newsgroups_test_original.target

##############################
### Prepare Training Data 
##############################
train_tokens = []  # list of token lists for gensim Doc2Vec
train_text = [] # list of document strings for sklearn TF-IDF
labels = []  # use filenames as labels
for doc in newsgroups_train_original.data:
    text_string = doc
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    train_tokens.append(tokens)
    train_text.append(text_string)
print('\nNumber of training documents:',
	len(train_text))	
print('\nFirst item after text preprocessing, train_text[0]\n', 
	train_text[0])
print('\nNumber of training token lists:',
	len(train_tokens))	
print('\nFirst list of tokens after text preprocessing, train_tokens[0]\n', 
	train_tokens[0])

##############################
### Prepare Test Data 
##############################
test_tokens = []  # list of token lists for gensim Doc2Vec
test_text = [] # list of document strings for sklearn TF-IDF
labels = []  # use filenames as labels
for doc in newsgroups_test_original.data:
    text_string = doc
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    test_tokens.append(tokens)
    test_text.append(text_string)
print('\nNumber of testing documents:',
	len(test_text))	
print('\nFirst item after text preprocessing, test_text[0]\n', 
	test_text[0])
print('\nNumber of testing token lists:',
	len(test_tokens))	
print('\nFirst list of tokens after text preprocessing, test_tokens[0]\n', 
	test_tokens[0])

##############################
### TF-IDF Vectorization
##############################
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)
print('\nTFIDF vectorization. . .')
print('\nTraining tfidf_vectors_training.shape:', tfidf_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT tfidf_vectorizer.fit_transform
tfidf_vectors_test = tfidf_vectorizer.transform(test_text)
print('\nTest tfidf_vectors_test.shape:', tfidf_vectors_test.shape)
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
tfidf_clf.fit(tfidf_vectors, train_target)
tfidf_pred = tfidf_clf.predict(tfidf_vectors_test)  # evaluate on test set
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))

##############################
### Count Vectorization
##############################
count_vectorizer = CountVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    max_features = VECTOR_LENGTH)
count_vectors = count_vectorizer.fit_transform(train_text)
print('\ncount vectorization. . .')
print('\nTraining count_vectors_training.shape:', count_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use count_vectorizer.transform, NOT count_vectorizer.fit_transform
count_vectors_test = count_vectorizer.transform(test_text)
print('\nTest count_vectors_test.shape:', count_vectors_test.shape)
count_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
count_clf.fit(count_vectors, train_target)
count_pred = count_clf.predict(count_vectors_test)  # evaluate on test set
print('\nCount/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, count_pred, average='macro'), 3))

##############################
### Hashing Vectorization
##############################
hashing_vectorizer = HashingVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    n_features = VECTOR_LENGTH)
hashing_vectors = hashing_vectorizer.fit_transform(train_text)
print('\ncount vectorization. . .')
print('\nTraining hashing_vectors_training.shape:', hashing_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use hashing_vectorizer.transform, NOT hashing_vectorizer.fit_transform
hashing_vectors_test = hashing_vectorizer.transform(test_text)
print('\nTest hashing_vectors_test.shape:', hashing_vectors_test.shape)
hashing_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
hashing_clf.fit(hashing_vectors, train_target)
hashing_pred = hashing_clf.predict(hashing_vectors_test)  # evaluate on test set
print('\nHashing/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, hashing_pred, average='macro'), 3))

###########################################
### Doc2Vec Vectorization (50 dimensions)
###########################################
# doc2vec paper:  https://cs.stanford.edu/~quocle/paragraph_vector.pdf
#     has a neural net with 1 hidden layer and 50 units/nodes
# documentation at https://radimrehurek.com/gensim/models/doc2vec.html
# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
# tutorial on GitHub: 
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
print('\nBegin Doc2Vec Work')
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)

train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_tokens)]
# print('train_corpus[:2]:', train_corpus[:2])

# Instantiate a Doc2Vec model with a vector size with 50 words 
# and iterating over the training corpus 40 times. 
# Set the minimum word count to 2 in order to discard words 
# with very few occurrences. 
# window (int, optional) – The maximum distance between the 
# current and predicted word within a sentence.
print("\nWorking on Doc2Vec vectorization, dimension 50")
model_50 = Doc2Vec(train_corpus, vector_size = 50, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_50.train(train_corpus, total_examples = model_50.corpus_count, 
	epochs = model_50.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_50_vectors = np.zeros((len(train_tokens), 50)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_50_vectors[i,] = model_50.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_50_vectors.shape:', doc2vec_50_vectors.shape)
# print('doc2vec_50_vectors[:2]:', doc2vec_50_vectors[:2])

# vectorization for the test set
doc2vec_50_vectors_test = np.zeros((len(test_tokens), 50)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_50_vectors_test[i,] = model_50.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_50_vectors_test.shape:', doc2vec_50_vectors_test.shape)

doc2vec_50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_50_clf.fit(doc2vec_50_vectors, train_target) # fit model on training set
doc2vec_50_pred = doc2vec_50_clf.predict(doc2vec_50_vectors_test)  # evaluate on test set
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)) 

###########################################
### Doc2Vec Vectorization (100 dimensions)
###########################################
print("\nWorking on Doc2Vec vectorization, dimension 100")
model_100 = Doc2Vec(train_corpus, vector_size = 100, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_100.train(train_corpus, total_examples = model_100.corpus_count, 
	epochs = model_100.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_100_vectors = np.zeros((len(train_tokens), 100)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_100_vectors[i,] = model_100.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_100_vectors.shape:', doc2vec_100_vectors.shape)
# print('doc2vec_100_vectors[:2]:', doc2vec_100_vectors[:2])

# vectorization for the test set
doc2vec_100_vectors_test = np.zeros((len(test_tokens), 100)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_100_vectors_test[i,] = model_100.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_100_vectors_test.shape:', doc2vec_100_vectors_test.shape)

doc2vec_100_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_100_clf.fit(doc2vec_100_vectors, train_target) # fit model on training set
doc2vec_100_pred = doc2vec_100_clf.predict(doc2vec_100_vectors_test)  # evaluate on test set
print('\nDoc2Vec_100/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3)) 

###########################################
### Doc2Vec Vectorization (200 dimensions)
###########################################
print("\nWorking on Doc2Vec vectorization, dimension 200")
model_200 = Doc2Vec(train_corpus, vector_size = 200, window = 4, 
	min_count = 2, workers = cores, epochs = 40)

model_200.train(train_corpus, total_examples = model_200.corpus_count, 
	epochs = model_200.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_200_vectors = np.zeros((len(train_tokens), 200)) # initialize numpy array
for i in range(0, len(train_tokens)):
    doc2vec_200_vectors[i,] = model_200.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_200_vectors.shape:', doc2vec_200_vectors.shape)
# print('doc2vec_200_vectors[:2]:', doc2vec_200_vectors[:2])

# vectorization for the test set
doc2vec_200_vectors_test = np.zeros((len(test_tokens), 200)) # initialize numpy array
for i in range(0, len(test_tokens)):
    doc2vec_200_vectors_test[i,] = model_200.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_200_vectors_test.shape:', doc2vec_200_vectors_test.shape)

doc2vec_200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, 
	random_state = SET_RANDOM)
doc2vec_200_clf.fit(doc2vec_200_vectors, train_target) # fit model on training set
doc2vec_200_pred = doc2vec_200_clf.predict(doc2vec_200_vectors_test)  # evaluate on test set
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)) 

print('\n\n------------------------------------------------------------------------')
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))
print('\nCount/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, count_pred, average='macro'), 3))
print('\nHashing/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, hashing_pred, average='macro'), 3))
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)) 
print('\nDoc2Vec_100/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3))   
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)) 
print('\n------------------------------------------------------------------------')