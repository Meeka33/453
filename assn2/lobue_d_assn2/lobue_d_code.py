
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

corpus_directory = '/home/meeka/Desktop/NU/453/assn2/philosophy/corpus'

from os import listdir
import re
import string
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

###############################################################################
#Define classification groups for eventual classification analysis
#create 6 vocabulary sets for class divisions based on common vocab words
###############################################################################


logic=['logic', 'logical', 'logics', 'syllogism', 'syllogisms', 'model']
mathematics=['mathematics','mathematical', 'number', 'set', 'sets', 'probability','probabilities', 'proof']
language=['language', 'linguistic', 'sentences', 'sentence', 'proposition', 'propositions', 'verb', 'verbs',
         'discourse', 'word', 'words']
mind=['cognition', 'cognitive', 'consciousness', 'thought', 'thoughts','knowledge', 'know', 'mental',
        'perception', 'neural', 'brain', 'mind', 'selfknowledge']
ontology=['objects', 'object', 'truth', 'abstract', 'abstraction', 'phenomenal', 'phenomenology',
             'representation', 'representational', 'representations', 'experience', 'experiences']
ethics=['ethics', 'ethical', 'moral', 'morality', 'religion']

top_vocab=[]

def merge_list(group):
    for word in group:
        top_vocab.append(word)

merge_list(logic)
merge_list(mathematics)
merge_list(language)
merge_list(mind)
merge_list(ontology)
merge_list(ethics)

len(top_vocab)


###############################################################################
#load individual corpus docs
#Create dict of entire corpus
#This will be used to process and align categories, top vocab terms with docs
###############################################################################

def load_doc(filename):
    file=open(filename, 'r')
    text=file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens=doc.split()
    tokens=[word.lower() for word in tokens]
    re_punc=re.compile('[%s]'% re.escape(string.punctuation))
    tokens=[re_punc.sub('',w) for w in tokens]
    tokens=[word for word in tokens if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if not word in stop_words]
    tokens=[word for word in tokens if len(word)>2]
    return tokens

def process_docs(directory):
    for filename in listdir(directory):
        path=directory+'/'+filename
        doc=load_doc(path)
        tokens=clean_doc(doc)

        #process lists, counters, dicts:
        vocab.update(tokens)
        wordcount=Counter(tokens)
        corpus_dict_top5[filename]=wordcount.most_common(5)
        line= ' '.join(tokens)
        corpus_dict_sent[filename]=line
        vocab_tokens=[word for word in tokens if word in top_vocab]
        vocabcount=Counter(vocab_tokens)
        corpus_vdict[filename]=vocabcount

def save_list(lines, filename):
    data='\n'.join(lines)
    file=open(filename, 'w')
    file.write(data)
    file.close()

vocab=Counter()

#Top 5 words in each document for categorization
corpus_dict_top5={}
#Dict with top wordcounts for top vocab 6 group categorization list
corpus_vdict={}
#Dict with full length sentences
corpus_dict_sent={}

process_docs(corpus_directory)
print('starting corpus document size: ', len(corpus_dict_sent))

min_occurrence=50
vocab=[k for k,c in vocab.items() if c >= min_occurrence]
save_list(vocab, 'vocab.txt')
print('Full corpus filtered vocab size: %d' % len(vocab))


###############################################################################
#create subset of docs for class labels based on category specific vocabulary
#merge with main corpus to align docs with class labels
#concatenate word within category with corresponding usage frequency
###############################################################################

def group_dfs(group):
    group_set={}
    for k,v in corpus_dict_top5.items():
        for name, count in v:
            if name in group:
                group_set[k]=v
    dkeys=[]
    dvals=[]
    for x,y in group_set.items():
        dkeys.append(x)
        dv=[]
        for item in y:
            dv.append(str(item[0]+'-'+str(item[1])))
        dvals.append(dv)
    headers=[]
    for x in range(1,6):
        label=str('word'+str(x))
        headers.append(label)
    newdf=pd.DataFrame(dvals, columns=headers, index=dkeys)
    return newdf

logic_df=group_dfs(logic)
mathematics_df=group_dfs(mathematics)
language_df=group_dfs(language)
mind_df=group_dfs(mind)
ontology_df=group_dfs(ontology)
ethics_df=group_dfs(ethics)

logic_df['class']='logic'
mathematics_df['class']='mathematics'
language_df['class']='language'
mind_df['class']='mind'
ontology_df['class']='ontology'
ethics_df['class']='ethics'

frames=[logic_df, mathematics_df, language_df, mind_df, ontology_df, ethics_df]
full_df=pd.concat(frames)
full_df=full_df.reset_index()
full_df=full_df.rename(columns={"index":"document"})
full_df.to_csv('full_concat.csv')

print('Docs categorized into groups based on top 5 words in lists:')
print(full_df['class'].value_counts())


###############################################################################
#Some docs have vocabulary words from multiple lists
#Identify Dups, separate class assignment based on word list class
###############################################################################

dups=full_df.duplicated(subset=['document'])
df_dup=pd.concat([full_df['document'], dups],axis=1, join='inner')
df_dup=df_dup.rename(columns={0:'duplicate'})
df_dup=df_dup[df_dup.duplicate]

duplist=df_dup.document.to_list()
uniquedocs=full_df[~full_df.document.isin(duplist)]
dupdocs=full_df[full_df.document.isin(duplist)]

print('full df shape with dups: ', full_df.shape)
print('unique docs shape: ', uniquedocs.shape)
print('dups docs shape: ', dupdocs.shape)
print('\nValue counts of unique doc classes:')
print(uniquedocs['class'].value_counts())


###############################################################################
#Clean duplicate docs based on first word assignment to category
#word that is top ranked in category is assigned to that category
###############################################################################
docfilter=dupdocs.drop_duplicates(subset='document').copy(deep=True)
docfilter2=docfilter.copy(deep=True)
docfilter2[['word1w','word1c']]=docfilter2.word1.str.split("-",expand=True)

keepcols=['document','word1w']
primaryword=docfilter2.filter(items=keepcols, axis=1).copy(deep=True)

NaN=np.nan
primaryword['class']=NaN

def firstword(group, name):
    for i in range(len(primaryword)):
        for word in group:
            if primaryword.iloc[i,1]==word:
                primaryword.iloc[i,2]=name

firstword(logic, 'logic')
firstword(mathematics, 'mathematics')
firstword(language, 'language')
firstword(mind, 'mind')
firstword(ontology, 'ontology')
firstword(ethics, 'ethics')

primaryword.dropna(subset=['class'], inplace=True)
docfilter=docfilter.drop(columns=['class'])
uniquefiltered=pd.concat([docfilter, primaryword['class']],axis=1, join='inner')
print('additional unique filtered docs to include: ', uniquefiltered.shape)

frames=[uniquedocs, uniquefiltered]
final_corpus_df=pd.concat(frames)
print('final corpus shape: ', final_corpus_df.shape)
print('final corpus counts: \n')
print(uniquedocs['class'].value_counts())



###############################################################################
#Approach 1 Summary EDA Analyst Judgment
###############################################################################

final_texts=final_corpus_df['document']
final_texts=final_texts.to_list()

final_vcorpus={}

for k,v in corpus_vdict.items():
    for word in final_texts:
        if k == word:
            final_vcorpus[k]=v

vocab_matrix=pd.DataFrame.from_dict(final_vcorpus, orient='index')
vocab_matrix.to_csv('vocab_matrix_analyst.csv')
vocab_matrix.iloc[:10,:10]


###############################################################################
#Approach 2 Summary EDA TF-IDF
###############################################################################

final_corpus=[]
final_labels=[]

for k,v in corpus_dict_sent.items():
    for word in final_texts:
        if k == word:
            final_corpus.append(v)
            final_labels.append(k)

vectorizer=TfidfVectorizer(max_features=3000)
X=vectorizer.fit_transform(final_corpus)
print(X.shape)

feature_names=vectorizer.get_feature_names()
corpus_index=[n for n in final_corpus]
Tfidf_df_matrix=pd.DataFrame(X.todense(), index=final_labels, columns=feature_names)
Tfidf_df_matrix.T.to_csv('tfidf_vocab_matrix_full.csv')

Tfidf_df_matrix_topVocab=Tfidf_df_matrix[top_vocab]
Tfidf_df_matrix_topVocab.iloc[:10,:10]
Tfidf_df_matrix_topVocab.to_csv('Tfidf_matrix_topVocab.csv')


###############################################################################
#extract data from overall corpus and split into train test for modeling
###############################################################################

doc_class_df=final_corpus_df[['document','class']]
final_texts=doc_class_df.values.tolist()

final_corpus_dict={}
final_labels_dict={}
final_analysis_dict={}

for k,v in corpus_dict_sent.items():
    for item in final_texts:
        if k == item[0]:
            final_corpus_dict[k]=v
            final_labels_dict[k]=item[1]
            final_analysis_dict[v]=item[1]

X=list(final_analysis_dict.keys())
y=list(final_analysis_dict.values())
print('items in X set: %d and items in y set: %d' %(len(X), len(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42)



###############################################################################
#Analysis 1: Best Word Judgment
###############################################################################

analyst_vectorizer=CountVectorizer(vocabulary=top_vocab)
X_train_analyst=analyst_vectorizer.fit_transform(X_train)
X_test_analyst=analyst_vectorizer.fit_transform(X_test)

#Analyst Dataframe
X_train_analyst_df=pd.DataFrame(X_train_analyst.todense(), columns=analyst_vectorizer.get_feature_names(), index=y_train)
X_train_analyst_df.head(5)

#Analysis 1.5: Full corpus
###############################################################################

analyst_vectorizer2=CountVectorizer(vocabulary=vocab)
X_train_analyst2=analyst_vectorizer2.fit_transform(X_train)
X_test_analyst2=analyst_vectorizer2.fit_transform(X_test)

#Analyst 2 Dataframe
X_train_analyst_df2=pd.DataFrame(X_train_analyst2.todense(), columns=analyst_vectorizer2.get_feature_names(), index=y_train)
X_train_analyst_df2.head(5)

#Analysis 2: Tf-Idf Dataframe
###############################################################################

#tfidf_vectorizer=TfidfVectorizer(max_features=1000)
tfidf_vectorizer=TfidfVectorizer(vocabulary=vocab)
X_train_tfidf=tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf=tfidf_vectorizer.fit_transform(X_test)

X_train_tfidf_df=pd.DataFrame(X_train_tfidf.todense(), columns=tfidf_vectorizer.get_feature_names(), index=y_train)
X_train_tfidf_df.head(5)


#Analysis 3: Doc2Vec
###############################################################################

def tokenize_docs(X):
    word_tokens=[]
    for doc in X:
        tokens=doc.split()
        word_tokens.append(tokens)
    return word_tokens

X_train_tokens=tokenize_docs(X_train)
X_test_tokens=tokenize_docs(X_test)

#50 Dim
######################################
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train_tokens)]
model_50dim = Doc2Vec(documents, vector_size=50, window=4, min_count=2, epochs=50)
model_50dim.train(documents, total_examples = model_50dim.corpus_count, epochs = model_50dim.epochs)

#Vectorize Training Set:
doc2vec_50_vectors = np.zeros((len(X_train_tokens), 50))
for i in range(0, len(X_train_tokens)):
    doc2vec_50_vectors[i,] = model_50dim.infer_vector(X_train_tokens[i]).transpose()
print(doc2vec_50_vectors.shape)

#Vectorize Test Set:
doc2vec_50_vectors_test = np.zeros((len(X_test_tokens), 50))
for i in range(0, len(X_test_tokens)):
    doc2vec_50_vectors_test[i,] = model_50dim.infer_vector(X_test_tokens[i]).transpose()
print(doc2vec_50_vectors_test.shape)

#150 Dim
######################################

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train_tokens)]
model_150dim = Doc2Vec(documents, vector_size=150, window=4, min_count=2, epochs=50)
model_150dim.train(documents, total_examples = model_150dim.corpus_count, epochs = model_150dim.epochs)

#Vectorize Training Set:
doc2vec_150_vectors = np.zeros((len(X_train_tokens), 150))
for i in range(0, len(X_train_tokens)):
    doc2vec_150_vectors[i,] = model_150dim.infer_vector(X_train_tokens[i]).transpose()
print(doc2vec_150_vectors.shape)

#Vectorize Test Set:
doc2vec_150_vectors_test = np.zeros((len(X_test_tokens), 150))
for i in range(0, len(X_test_tokens)):
    doc2vec_150_vectors_test[i,] = model_150dim.infer_vector(X_test_tokens[i]).transpose()
print(doc2vec_150_vectors_test.shape)


#200 Dim
######################################

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train_tokens)]
model_200dim = Doc2Vec(documents, vector_size=200, window=4, min_count=2, epochs=50)
model_200dim.train(documents, total_examples = model_150dim.corpus_count, epochs = model_150dim.epochs)

#Vectorize Training Set:
doc2vec_200_vectors = np.zeros((len(X_train_tokens), 200))
for i in range(0, len(X_train_tokens)):
    doc2vec_200_vectors[i,] = model_200dim.infer_vector(X_train_tokens[i]).transpose()
print(doc2vec_200_vectors.shape)

#Vectorize Test Set:
doc2vec_200_vectors_test = np.zeros((len(X_test_tokens), 200))
for i in range(0, len(X_test_tokens)):
    doc2vec_200_vectors_test[i,] = model_200dim.infer_vector(X_test_tokens[i]).transpose()
print(doc2vec_200_vectors_test.shape)



###############################################################################
###############################################################################
#Modeling For Predictions
###############################################################################
###############################################################################


#Analyst Judgement #1
###############################################################################

count_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
count_clf.fit(X_train_analyst, y_train)
count_pred = count_clf.predict(X_test_analyst)
print('\nAnalyst/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, count_pred, average='macro'), 3))

#Full Document Vectorization
###############################################################################
count2_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
count2_clf.fit(X_train_analyst2, y_train)
count2_pred = count2_clf.predict(X_test_analyst2)
print('\nFull Doc Vectorization/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, count2_pred, average='macro'), 3))

#Tf-Idf
###############################################################################
tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
tfidf_clf.fit(X_train_tfidf, y_train)
tfidf_pred = tfidf_clf.predict(X_test_tfidf)
print('\nTF-IDF/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, tfidf_pred, average='macro'), 3))

#Doc2Vec 50
###############################################################################
doc2vec_50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
doc2vec_50_clf.fit(doc2vec_50_vectors, y_train)
doc2vec_50_pred = doc2vec_50_clf.predict(doc2vec_50_vectors_test)
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, doc2vec_50_pred, average='macro'), 3))

#Doc2Vec 150
###############################################################################
doc2vec_150_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
doc2vec_150_clf.fit(doc2vec_150_vectors, y_train)
doc2vec_150_pred = doc2vec_150_clf.predict(doc2vec_150_vectors_test)
print('\nDoc2Vec_150/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, doc2vec_150_pred, average='macro'), 3))

#Doc2Vec 200
###############################################################################
doc2vec_200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
doc2vec_200_clf.fit(doc2vec_200_vectors, y_train)
doc2vec_200_pred = doc2vec_200_clf.predict(doc2vec_200_vectors_test)
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, doc2vec_200_pred, average='macro'), 3))


###############################################################################
#Factorial Design Comparisons
###############################################################################

#Analysis 2: Full Doc Vectorization 50
analyst_vectorizer50=CountVectorizer(max_features=50)
X_train_analyst50=analyst_vectorizer50.fit_transform(X_train)
X_test_analyst50=analyst_vectorizer50.fit_transform(X_test)

#Analysis 2: Full Doc Vectorization 150
analyst_vectorizer150=CountVectorizer(max_features=150)
X_train_analyst150=analyst_vectorizer150.fit_transform(X_train)
X_test_analyst150=analyst_vectorizer150.fit_transform(X_test)

#Analysis 2: Full Doc Vectorization 200
analyst_vectorizer200=CountVectorizer(max_features=200)
X_train_analyst200=analyst_vectorizer200.fit_transform(X_train)
X_test_analyst200=analyst_vectorizer200.fit_transform(X_test)

#tfidf_vectorizer 50
tfidf_vectorizer50=TfidfVectorizer(max_features=50)
X_train_tfidf50=tfidf_vectorizer50.fit_transform(X_train)
X_test_tfidf50=tfidf_vectorizer50.fit_transform(X_test)

#tfidf_vectorizer 150
tfidf_vectorizer150=TfidfVectorizer(max_features=150)
X_train_tfidf150=tfidf_vectorizer150.fit_transform(X_train)
X_test_tfidf150=tfidf_vectorizer150.fit_transform(X_test)

#tfidf_vectorizer 20
tfidf_vectorizer200=TfidfVectorizer(max_features=200)
X_train_tfidf200=tfidf_vectorizer200.fit_transform(X_train)
X_test_tfidf200=tfidf_vectorizer200.fit_transform(X_test)

################################################

#Analysis 2: Full Doc Vectorization 50
count50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)

count50_clf.fit(X_train_analyst50, y_train)
count50_pred = count50_clf.predict(X_test_analyst50)
print('\nFull Doc 50 Vectorization/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, count50_pred, average='macro'), 3))

#Analysis 2: Full Doc Vectorization 150
count150_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
count150_clf.fit(X_train_analyst150, y_train)
count150_pred = count150_clf.predict(X_test_analyst150)
print('\nFull Doc 150 Vectorization/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, count150_pred, average='macro'), 3))

#Analysis 2: Full Doc Vectorization 200
count200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
count200_clf.fit(X_train_analyst200, y_train)
count200_pred = count200_clf.predict(X_test_analyst200)
print('\nFull Doc 200 Vectorization/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, count200_pred, average='macro'), 3))

################################################

#Tf-Idf 50
tfidf50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
tfidf50_clf.fit(X_train_tfidf50, y_train)
tfidf50_pred = tfidf50_clf.predict(X_test_tfidf50)
print('\nTF-IDF 50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, tfidf50_pred, average='macro'), 3))

#Tf-Idf 150
tfidf150_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
tfidf150_clf.fit(X_train_tfidf150, y_train)
tfidf150_pred = tfidf150_clf.predict(X_test_tfidf150)
print('\nTF-IDF 150/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, tfidf150_pred, average='macro'), 3))

#Tf-Idf 200
tfidf200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42)
tfidf200_clf.fit(X_train_tfidf200, y_train)
tfidf200_pred = tfidf200_clf.predict(X_test_tfidf200)
print('\nTF-IDF 200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(y_test, tfidf200_pred, average='macro'), 3))


###############################################################################
###############################################################################
#Convolutional Neural Network Model
###############################################################################
###############################################################################

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def clean_doc(corpus, vocab):
    slim_doc=[]
    for doc in corpus:
        tokens = doc.split()
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
        slim_doc.append(tokens)
    return slim_doc

X_train_slim=clean_doc(X_train, vocab)
X_test_slim=clean_doc(X_test, vocab)

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding= 'post')
    return padded

def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(128, 8, activation= 'relu' ))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation= 'relu'))
    model.add(Dense(6, activation= 'softmax'))
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
    model.summary()
    return model

tokenizer = create_tokenizer(X_train_slim)
vocab_size = len(tokenizer.word_index) + 1
print( 'Vocabulary size: %d ' % vocab_size)

max_length = max([len(s.split()) for s in X_train_slim])
print( 'Maximum length: %d ' % max_length)

Xtrain_s = encode_docs(tokenizer, max_length, X_train_slim)
Xtest_s = encode_docs(tokenizer, max_length, X_test_slim)

le = LabelEncoder()
ytrain = le.fit_transform(y_train)
ytrain=to_categorical(ytrain)

ytest=le.fit_transform(y_test)
ytest=to_categorical(ytest)

basic_model = define_model(vocab_size, max_length)
basic_model.fit(Xtrain_s, ytrain, epochs=10, verbose=2)


###############################################################################
#results
###############################################################################
from sklearn.metrics import classification_report

_, acc = basic_model.evaluate(Xtest_s, ytest, verbose=0)
print(' Test Accuracy: %.2f ' % (acc*100))


le = LabelEncoder()
ytest = le.fit_transform(y_test)

y_pred = basic_model.predict(Xtest_s, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(ytest, y_pred_bool))
print('Encoded label order 0:5: ', list(le.inverse_transform([0,1,2,3,4,5])))
