import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
import string
#from morphological_analysis import lemma
import nltk
import csv
from nltk.corpus import wordnet
import enchant
from collections import Counter
from  nltk.stem.wordnet import WordNetLemmatizer

word_verification = enchant.Dict("en_US")

'''
def read_book(path):
    with open(path, encoding="ISO-8859-1") as file:
        data = file.read()
        return data

def procesing(content):
    content = content.lower()
    content = " ".join(content.split())
    for c in string.punctuation:
        content = content.replace(c, "")
    content = ''.join([i for i in content if not i.isdigit()])
    words = word_tokenize(content, language='portuguese')
    words = [lemma(word) for word in words]
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    str_words = ' '.join(words)
    str_filtered = ' '.join(filtered_words)
    return str_words, len(words),str_filtered, len(filtered_words)


books = os.listdir('books/')
books.remove('.DS_Store')

stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords = {i : x for x, i in enumerate(stopwords)}


container = dict()
headers = ['book_id', 'category' ,'complete_content', 'words_complete', 'filtered_content', 'words_filtered']

all_columns = []
index = 1
count = 1
for category in books:
    books_per_category = os.listdir('books/' + category)
    for book in books_per_category:
        bp = 'books/' + category + '/' +book
        content = read_book(bp).split()
        content = content[300:]
        content = content[:len(content) - 3200]
        size = len(content)
        if size>4000:
            print(index,count, book)
            count+=1
            content = ' '.join(content)
            content, len_content, content_filtered, len_filtered = procesing(content)
            book_id = book[:book.rfind('.')]
            row = [book_id, category ,content, len_content, content_filtered, len_filtered]
            all_columns.append(row)
        index+=1


with open('books.csv', mode='w') as myFile:
    writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    for row in all_columns:
        writer.writerow(row)
'''

import pickle
def load_data_from_disk(file):
    with open(file, 'rb') as fid:
        data = pickle.load(fid)
    return data

stop_list = load_data_from_disk('extras/stopwords.pk')

def get_pos(word):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
    pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
    pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
    pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]

def process(text):
    content = text.lower()
    for c in string.punctuation:
        content = content.replace(c, ' ')
    content = ''.join([i for i in content if not i.isdigit()])
    content = word_tokenize(content)
    words = [WordNetLemmatizer().lemmatize(word, get_pos(word)) for word in content]
    filtered_words = [word for word in words if word not in stop_list and len(word) > 2]
    str_words = ' '.join(words)
    str_filtered = ' '.join(filtered_words)
    return str_words, len(words), str_filtered, len(filtered_words)



test_db = pd.read_csv('../databases/english.csv')
other_bd = pd.read_csv('../databases/books_authorship.csv')

#test_db.info()
print()
#other_bd.info()

#authors = list(set(list(test_db['label'])))
#content = list(test_db['words'])
'''
headers = ['author', 'book_id', 'complete_content', 'words_complete', 'filtered_content', 'words_filtered']
with open('books_english.csv', mode='w') as myFile:
    writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    count = 1
    for index, row in test_db.iterrows():
        book_id = row['name']
        author = row['label']
        text = row['words']
        print(count, book_id, author)
        content, len_content, content_filtered, len_filtered = process(text)
        count+=1
        row = [author, book_id, content, len_content, content_filtered, len_filtered]
        writer.writerow(row)

'''

books = pd.read_csv('books_english.csv')
books.info()

sizes = list(books['words_complete'])
print(max(sizes), min(sizes), np.mean(sizes))

sizes2 = list(books['words_filtered'])
print(max(sizes2), min(sizes2), np.mean(sizes2))