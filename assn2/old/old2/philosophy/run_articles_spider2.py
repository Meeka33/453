
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

page_dirname = 'corpus'
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


from collections import Counter
from bs4 import BeautifulSoup
import string
import re
import dataprocesses as dp


#Next, create an overall vocabulary of most frequently used words after tokenizing
#Start by creating a vocabulary to pull from

def add_doc_to_vocab(filename, vocab):
    doc=dp.load_doc(filename)
    tokens=dp.clean_doc(doc)
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

def doc_to_line(filename, vocab):
    doc=dp.load_doc(filename)
    tokens=dp.clean_doc(doc)
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
vocab=dp.load_doc(vocab_filename)
vocab=set(vocab.split())
docs, labels=load_clean_dataset(vocab)

print('total number of documents: ', len(docs))
print('\n'*2)


###############################################################
#Create method of document Classification
#Build TF-IDF Using Sklearn and incorporate function that retuns Cosine Similarity

#Enter query from terminal:
begin=input("Would you like to search the philosophy of languge corpus? (y/n)")
if begin=='y':
    dp.QA(docs, labels)
    ask_again=input("Would you like to search again? (y/n)")
    if ask_again=='y':
        dp.QA(docs, labels)
    else:
        pass
else:
    pass
