
import scrapy
import os

#Delete any prior .jl lines instead of appending:
if os.path.exists('items.jl'):
    os.remove('items.jl')
else:
    print("File does not yet exist, generating new output file")

# make directory for storing complete html code for web page
page_dirname = 'philanguage'
if not os.path.exists(page_dirname):
	os.makedirs(page_dirname)

# function for walking and printing directory structure
def list_all(current_directory):
    for root, dirs, files in os.walk(current_directory):
        level = root.replace(current_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# examine the directory structure
current_directory = os.getcwd()
list_all(current_directory)

# list the avaliable spiders
print('\nScrapy spider names:\n')
os.system('scrapy list')

# for JSON lines we use this command
os.system('scrapy crawl articles-spider -o items.jl')
print('\nJSON lines written to items.jl\n')

####################################################################################################
####################################################################################################

import string
import re
from collections import Counter
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


# Next, process Data since original web pages have a wide range of topics covered
# Start by breaking apart documents into sub sections, since overall articles can vary widely in topics:
# This will allow individual documents to be more focused

page_dirname = 'corpus'
if not os.path.exists(page_dirname):
    os.makedirs(page_dirname)

def process_htmldocs(directory):
    for filename in os.listdir(directory):
        path=directory + '/' + filename
        load_htmldoc(path)

def load_htmldoc(filename):
    file=open(filename, 'r')
    content=file.read()
    file.close()
    webtext=BeautifulSoup(content, 'html.parser')
    pagetext=webtext.find(id="main-text")
    
    #Identify sub headers for topic sections
    #subject=[headers.text for headers in webtext.find_all('h1')]
    subject=[headers.text.encode('utf-8').decode('ascii', 'ignore') for headers in webtext.find_all('h1')]
    topics=[]
    for headers in pagetext.find_all('h2'):
        headers=headers.text.encode('utf-8').decode('ascii', 'ignore')
        removal=string.punctuation
        for item in removal:
            headers=headers.replace(item, '')
        # encoded_string=headers.encode("ascii", "ignore")
        # headers=encoded_string.decode() 
        topics.append(subject[0]+'_'+headers)

    # topics=[item.encode("ascii", "ignore") for item in topics]
    # topics=[encoded_string.decode() for encoded_string in topics]
        
    #Split HTML sections by header classes
    fullbody=str(pagetext)
    body_split = re.split('(<h2>|</h2>)', fullbody)
    collection=[]
    for section in body_split:
        clean=BeautifulSoup(section, 'html.parser')
        paragraphs=clean.select('p')
        if paragraphs:
            collection.append(paragraphs)
   
    #split and clean each section of text
    counter=0
    for section in collection:
        try:
            text=''
            for x in range(len(section)):
                sent=str(section[x].text)
                sent=sent.strip().replace('\n', ' ')
                sent=sent.strip().replace('<p>', ' ')
                sent=sent.strip().replace('</p>', ' ')
                sent=sent.strip().replace('<em>', ' ')
                sent=sent.strip().replace('</em>', ' ')
                text=text +' '+ sent
            filename = page_dirname + '/' + topics[counter]+'.txt'
            save_list(text, filename)
            counter=counter+1
        except:
            pass

def save_list(lines, filename):
    file=open(filename, 'w')
    file.write(lines)
    file.close()
    
process_htmldocs('philanguage')


#Next, create an overall vocabulary of most frequently used words after tokenizing
#Start by creating a vocabulary to pull from

def load_txtdoc(filename):
    file=open(filename, 'r')
    text=file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens=doc.split()
    tokens=[word.lower() for word in tokens]
    re_punc=re.compile('[%s]' % re.escape(string.punctuation))
    tokens=[re_punc.sub('',w) for w in tokens]
    tokens=[word for word in tokens if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if not word in stop_words]
    tokens=[word for word in tokens if len(word)>1]
    return tokens

def add_doc_to_vocab(filename, vocab):
    doc=load_txtdoc(filename)
    tokens=clean_doc(doc)
    vocab.update(tokens)

def process_txtdocs(directory, vocab):
    for filename in os.listdir(directory):
        path=directory + '/' + filename
        add_doc_to_vocab(path, vocab)

def save_voc_list(lines, filename):
    data='\n'.join(lines)
    file=open(filename, 'w')
    file.write(data)
    file.close()
        
vocab=Counter()
process_txtdocs('corpus', vocab)
print('length of vocabulary:', len(vocab))

min_occurrence=3
tokens=[k for k,c in vocab.items() if c>min_occurrence]
print('number of tokens: ', len(tokens))

save_voc_list(tokens, 'vocab.txt')


#Now data can be processed for utilization
#Now create bag of words: Import articles, clean, filter words only in vocab defined above, convert to string

def load_doc(filename):
    file=open(filename, 'r')
    text=file.read()
    file.close()
    return text

def clean_doc2(doc):
    tokens=doc.split()
    tokens=[word.lower() for word in tokens]
    re_punc=re.compile('[%s]' % re.escape(string.punctuation))
    tokens=[re_punc.sub('',w) for w in tokens]
    tokens=[word for word in tokens if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if not word in stop_words]
    tokens=[word for word in tokens if len(word)>1]
    return tokens

def doc_to_line(filename, vocab):
    doc=load_doc(filename)
    tokens=clean_doc2(doc)
    tokens=[w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs(directory, vocab):
    lines=list()
    titles=list()
    for filename in os.listdir(directory):
        path=directory + '/' + filename
        line=doc_to_line(path, vocab)
        lines.append(line)
        titles.append(filename)
    return lines, titles

def load_clean_dataset(vocab):
    docs,labels=process_docs('corpus', vocab)
    return docs, labels

vocab_filename='vocab.txt'
vocab=load_doc(vocab_filename)
vocab=set(vocab.split())
docs, labels=load_clean_dataset(vocab)

print('total number of documents: ', len(docs))


###############################################################
#Create method of document Classification
#Build TF-IDF Using Sklearn and incorporate function that retuns Cosine Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def Doc_query(query):
    docTFIDF = TfidfVectorizer().fit_transform(docs)
    queryTFIDF = TfidfVectorizer().fit(docs)
    queryTFIDF = queryTFIDF.transform([query])

    cosineSimilarities = cosine_similarity(queryTFIDF, docTFIDF).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-5:-1]
    
    print("The top document hit is: \n", (labels[related_docs_indices[0]]).rstrip('.txt'))
    print("\nThe relevant search identifiers are:")
    print(Counter(docs[related_docs_indices[0]].split()).most_common(5))
    answer=input("Would you like to see more detail about this article? (y/n) \n")
    if answer=='y':
        print("Full text excerpt is as follows: \n")
        path= 'corpus/'+ labels[related_docs_indices[0]]
        f=open(path, 'r')
        for line in f:
            print(line)
        f.close()
    else:
        pass
    
    answer=input("Would you like to see another document? (y/n) \n")
    if answer=='y':
        print("\nThe next document hit is: \n", (labels[related_docs_indices[1]]).rstrip('.txt'))
        print("\nThe relevant search identifiers are:")
        print(Counter(docs[related_docs_indices[1]].split()).most_common(5))
        answer=input("Would you like to see more detail about this article? (y/n) \n")
        if answer=='y':
            print("Full text excerpt is as follows: \n")
            path= 'corpus/'+ labels[related_docs_indices[1]]
            f=open(path, 'r')
            for line in f:
                print(line)
            f.close()
        else:
            pass
    else:
        pass


#Enter query from terminal:

begin=input("Would you like to search the philosophy of languge corpus? (y/n)")
if begin=='y':
    query=input("What would you like to learn about today?")
    Doc_query(query)

    ask_again=input("Would you like to search again? (y/n)")
    if ask_again=='y':
        query=input("What would you like to learn about today?")
        Doc_query(query)
    else:
        pass
else:
    pass

