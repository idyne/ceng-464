import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk.wsd import lesk
from nltk.tokenize import sent_tokenize
import string

spell = Speller()
sentence = open("txt_sentoken/neg/cv000_29416.txt", 'r').read()
words = word_tokenize(sentence)
corrected_sentence = ""
corrected_word_list = []
for wd in words:
    if wd not in string.punctuation:
        wd_c = spell(wd)
        if wd_c != wd:
            print(wd + " has been corrected to: " + wd_c)
            corrected_sentence = corrected_sentence + " " + wd_c
            corrected_word_list.append(wd_c)
        else:
            corrected_sentence = corrected_sentence + " " + wd
            corrected_word_list.append(wd)
    else:
        corrected_sentence = corrected_sentence + wd
        corrected_word_list.append(wd)

stop_words = stopwords.words('English')
corrected_word_list_without_stopwords = []
for wd in corrected_word_list:
    if wd not in stop_words:
        corrected_word_list_without_stopwords.append(wd)
stemmer = nltk.stem.PorterStemmer()
corrected_word_list_without_stopwords_stemmed = []
for wd in corrected_word_list_without_stopwords:
    corrected_word_list_without_stopwords_stemmed.append(stemmer.stem(wd))
lemmatizer = WordNetLemmatizer()
corrected_word_list_without_stopwords_lemmatized = []
for wd in corrected_word_list_without_stopwords:
    corrected_word_list_without_stopwords_lemmatized.append(lemmatizer.
                                                            lemmatize(wd))
for sentence in sent_tokenize(corrected_sentence):
    print(sentence)