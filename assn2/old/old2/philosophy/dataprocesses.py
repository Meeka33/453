
import string
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_doc(filename):
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

def Doc_query(query, docs):
    docTFIDF = TfidfVectorizer().fit_transform(docs)
    queryTFIDF = TfidfVectorizer().fit(docs)
    queryTFIDF = queryTFIDF.transform([query])

    cosineSimilarities = cosine_similarity(queryTFIDF, docTFIDF).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-5:-1]
    return related_docs_indices

def retreivetext(labels, related_docs_indices, docno):
    print("Full text excerpt is as follows: \n")
    path= 'corpus/'+ labels[related_docs_indices[docno]]
    f=open(path, 'r')
    for line in f:
        print(line)
    f.close()


def QA(docs, labels):
    query=input("What would you like to learn about today?")
    related_docs_indices=Doc_query(query, docs)
    print("\nThe top document hit is: \n", (labels[related_docs_indices[0]]).rstrip('.txt'))
    print("\nThe relevant search identifiers are:")
    print(Counter(docs[related_docs_indices[0]].split()).most_common(5), '\n')
    answer=input("\nWould you like to see more detail about this article? (y/n) \n")
    if answer=='y':
        retreivetext(labels,related_docs_indices,0)
    else:
        pass
    answer=input("Would you like to see another document? (y/n) \n")
    if answer=='y':
        print("\nThe next document hit is: \n", (labels[related_docs_indices[1]]).rstrip('.txt'))
        print("\nThe relevant search identifiers are:")
        print(Counter(docs[related_docs_indices[1]].split()).most_common(5), '\n')
        answer=input("\nWould you like to see more detail about this article? (y/n) \n")
        if answer=='y':
            retreivetext(labels,related_docs_indices,1)
        else:
            pass
    else:
        pass
