import pandas as pd
from string import punctuation
import nltk

nltk.download('tagsets')
from nltk.data import load

nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk import word_tokenize
from collections import Counter
import seaborn as sns
import contractions
import networkx as nx
from gensim.parsing.preprocessing import preprocess_string
tagdict = load('help/tagsets/upenn_tagset.pickle')
list(tagdict.keys())
data = pd.read_csv('data_ch2/data.csv', header=0)
pos_di = {}
for pos in list(tagdict.keys()):
    pos_di[pos] = []
for doc in data['text']:
    di = Counter([j for i, j in pos_tag(word_tokenize(doc))])
    for pos in list(tagdict.keys()):
        pos_di[pos].append(di[pos])
feature_df = pd.DataFrame(pos_di)
feature_df.head()
feature_df['num_of_unique_punctuations'] = data['text'].apply(lambda x:
                                                              len(set(x).intersection(set(punctuation))))
feature_df['num_of_unique_punctuations'].head()
feature_df['number_of_capital_words'] = data['text'].apply(
    lambda x: len([word for word in word_tokenize(str(x)) if word[0].isupper()]))
feature_df['number_of_capital_words'].head()
feature_df['number_of_small_words'] = data['text'].apply(
    lambda x: len([word for word in word_tokenize(str(x)) if word[0].islower()]))
feature_df['number_of_small_words'].head()
feature_df['number_of_alphabets'] = data['text'].apply(lambda x: len([ch
                                                                      for ch in str(x) if ch.isalpha()]))
feature_df['number_of_alphabets'].head()
feature_df['number_of_digits'] = data['text'].apply(lambda x: len([ch for
                                                                   ch in str(x) if ch.isdigit()]))
feature_df['number_of_digits'].head()
feature_df['number_of_words'] = data['text'].apply(lambda x: len(word_tokenize(str(x))))
feature_df['number_of_words'].head()
feature_df['number_of_white_spaces'] = data['text'].apply(lambda x:
                                                          len(str(x).split(' ')) - 1)
feature_df['number_of_white_spaces'].head()
feature_df.head()
