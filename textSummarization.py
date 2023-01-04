import pandas as pd
import nltk
import re
import contractions
import numpy as np
import zipfile
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
import networkx as nx

nltk.download('punkt')  # one time execution
pd.set_option('display.max_colwidth', 1000)
GLOVE_DIR = 'data/glove/'
GLOVE_ZIP = GLOVE_DIR + 'glove.6B.50d.zip'
zip_ref = zipfile.ZipFile(GLOVE_ZIP, 'r')
zip_ref.extractall(GLOVE_DIR)
zip_ref.close()


def load_glove_vectors(fn):
    print("Loading Glove Model")
    with open(fn, 'r', encoding='utf8') as glove_vector_file:
        model = {}
        for line in glove_vector_file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print("Loaded {} words".format(len(model)))
    return model


glove_vectors = load_glove_vectors('data/glove/glove.6B.50d.txt')
articles = pd.read_csv("data/tennis_articles_v4.csv")
print(articles.head(2))
nltk.download('stopwords')
stop_words = stopwords.words('english')
CLEAN_PATTERN = r'[^a-zA-z\s]'


def clean(word):
    return re.sub(CLEAN_PATTERN, '', word)


def clean_sentence(sentence):
    sentence = [clean(word) for word in sentence]
    return [word for word in sentence if word]


def clean_sentences(sentences):
    return [clean_sentence(sentence) for sentence in sentences]


def lower(sentence):
    return [word.lower() for word in sentence]


def remove_stopwords(sentence):
    words = [word for word in sentence if word not in stop_words]
    return [word for word in words if len(word) > 0]


def tokenize_words(sentences):
    return [word_tokenize(sentence) for sentence in sentences]


def fix_contractions(sentences):
    return [contractions.fix(sentence) for sentence in sentences]


articles['SentencesInArticle'] = articles.article_text.apply(sent_tokenize)
articles['WordsInSentences'] = articles.SentencesInArticle.apply(fix_contractions).apply(lower).apply(
    tokenize_words).apply(remove_stopwords).apply(clean_sentences)

articles = articles[['SentencesInArticle', 'WordsInSentences']]
articles.head(2)

articles.head(2)

VECTOR_SIZE = 50
EMPTY_VECTOR = np.zeros(VECTOR_SIZE)


def sentence_vector(sentence):
    print(type(sum([glove_vectors.get(word, EMPTY_VECTOR) for word in sentence])))
    result = sum([glove_vectors.get(word, EMPTY_VECTOR) for word in sentence]) / len(sentence)
    return result


def sentences_to_vectors(sentences):
    return [sentence_vector(sentence) for sentence in sentences]


articles['SentenceVector'] = articles.WordsInSentences.apply(sentences_to_vectors)


def similarity_matrix(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):

            element_i = sentence_vectors[i].reshape(1, VECTOR_SIZE)
            element_j = sentence_vectors[j].reshape(1, VECTOR_SIZE)
            sim_mat[i][j] = cosine_similarity(element_i, element_j)[0, 0]
    return sim_mat


articles['SimMatrix'] = articles.SentenceVector.apply(similarity_matrix)


def compute_graph(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    return scores


articles['Graph'] = articles.SimMatrix.apply(compute_graph)

articles.head(2)


def get_ranked_sentences(sentences, scores, n=3):
    top_scores = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences = [sentence for score, sentence in top_scores[:n]]
    return " ".join(top_n_sentences)


articles['Summary'] = articles.apply(lambda d: get_ranked_sentences(d.SentencesInArticle, d.Graph), axis=1)
