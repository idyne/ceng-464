# region Imports
import builtins
import string
import warnings
from multiprocessing import Process, freeze_support

import nltk
import pandas as pd
import seaborn as sns
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pylab import *
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report, \
    auc, roc_curve, mean_absolute_error
import nltk
import re
import contractions
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from fileIO import FileIO
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# endregion

# region Initialization
def initialize():
    warnings.filterwarnings('ignore')
    nltk.download('stopwords')
    nltk.download('tagsets')
    nltk.download('averaged_perceptron_tagger')
    sns.set()
    nltk.download('punkt')  # one time execution
    pd.set_option('display.max_colwidth', 1000)


# endregion

# region Read Files
def read_files():
    neg_texts = FileIO.read_recursively("txt_sentoken/neg/")
    pos_texts = FileIO.read_recursively("txt_sentoken/pos/")
    return neg_texts, pos_texts


def create_data_frame_of_texts(texts):
    sentiments = []
    for i in range(1000):
        sentiments.append(0)
    for i in range(1000):
        sentiments.append(1)
    data_dict = {"text": texts, "sentiment": sentiments}
    data = pd.DataFrame(data_dict)
    return data


# endregion

# region Part A
def clean_texts(data):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    stop_words = stop_words + list(string.printable)
    data['cleaned_text'] = data['text'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if
         word.lower() not in stop_words]))
    return data


# endregion

# region Part B
# region Part B.1
def extract_features(data):
    feature_df = pd.DataFrame({})
    # ii) number of punctations
    feature_df['num_of_unique_punctuations'] = data['text'].apply(
        lambda x: len(set(x).intersection(set(string.punctuation))))

    # iii) number of capital and small letter words
    feature_df['number_of_capital_words'] = data['text'].apply(
        lambda x: len([word for word in word_tokenize(str(x)) if word[0].isupper()]))

    feature_df['number_of_small_words'] = data['text'].apply(
        lambda x: len([word for word in word_tokenize(str(x)) if word[0].islower()]))

    # iv) number of alphabets
    feature_df['number_of_alphabets'] = data['text'].apply(lambda x: len([ch for ch in str(x) if ch.isalpha()]))

    # v) number of digits
    feature_df['number_of_digits'] = data['text'].apply(lambda x: len([ch for ch in str(x) if ch.isdigit()]))

    # vi) number of words
    feature_df['number_of_words'] = data['text'].apply(lambda x: len(word_tokenize(str(x))))

    # vii) number of white spaces for each sentence
    feature_df['number_of_white_spaces'] = data['text'].apply(lambda x: len(str(x).split(' ')) - 1)

    return feature_df


# endregion

# region Part B.2

def create_bag_of_words(data):
    bag_of_words_model = CountVectorizer()
    bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform(data['cleaned_text']).todense())
    bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
    return bag_of_word_df


def create_tfidf(data):
    tfidf_model = TfidfVectorizer()
    tfidf_df = pd.DataFrame(tfidf_model.fit_transform(data['cleaned_text']).todense())
    tfidf_df.columns = sorted(tfidf_model.vocabulary_)
    return tfidf_df


# endregion

# region Part B.3
def show_word_cloud(data, bag_of_word_df):
    from wordcloud import WordCloud, STOPWORDS
    word_frequencies = {}
    sorted_words_by_frequency = []
    for key in bag_of_word_df.keys():
        word_frequencies[key] = bag_of_word_df[key].values.sum()
    other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI', "n'"]
    STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=50, stopwords=stopwords,
                          min_font_size=10).generate_from_frequencies(word_frequencies)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# endregion

# region Part B.4

def clf_model(model_type, X_train, y_train, X_valid):
    model = model_type.fit(X_train, y_train)
    predicted_labels = model.predict(X_valid)
    predicted_probab = model.predict_proba(X_valid)[:, 1]
    return [predicted_labels, predicted_probab, model]


def reg_model(model_type, X_train, y_train, X_valid):
    # Fit the model to the training data
    model_type.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model_type.predict(X_valid)
    return y_pred


def reg_model_evaluation(actual_values, predicted_values):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rms = sqrt(mean_squared_error(actual_values, predicted_values))
    print('Root Mean Squared Error (RMSE) is:', rms)
    # Evaluate the model using the mean absolute error
    mae = mean_absolute_error(actual_values, predicted_values)
    print(f'Mean Absolute Error: {mae:.2f}')


def clf_model_evaluation(actual_values, predicted_values, predicted_probabilities):
    cfn_mat = confusion_matrix(actual_values, predicted_values)
    print("confusion matrix: \n", cfn_mat)
    print("\naccuracy: ", accuracy_score(actual_values, predicted_values))
    print("\nclassification report: \n", classification_report(actual_values, predicted_values))
    fpr, tpr, threshold = roc_curve(actual_values, predicted_probabilities)
    print('\nArea under ROC curve for validation set:', auc(fpr, tpr))
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rms = sqrt(mean_squared_error(actual_values, predicted_values))
    print('Root Mean Squared Error (RMSE) is:', rms)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label='Validation set AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.legend(loc='best')
    plt.show()


def do_kmeans_clustering(data, tfidf_df):
    kmeans = KMeans(n_clusters=13)
    kmeans.fit(tfidf_df)
    y_kmeans = kmeans.predict(tfidf_df)
    data['obtained_clusters'] = y_kmeans
    print(pd.crosstab(data['sentiment'].replace({0: 'neg', 1: 'pos'}), data['obtained_clusters'].replace(
        {0: 'cluster_1', 1: 'cluster_2', 2: 'cluster_3', 3: 'cluster_4'})))
    # Using Elbow method to obtain the number of clusters
    distortions = []
    K = range(1, 16)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(tfidf_df)
        distortions.append(
            sum(np.min(cdist(tfidf_df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / tfidf_df.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal number of clusters')
    plt.show()


def do_linear_regression(y_train, X_train, X_valid, y_valid):
    linreg = LinearRegression()
    result = reg_model(linreg, X_train, y_train, X_valid)
    reg_model_evaluation(y_valid, result)


def do_random_forest_regression(y_train, X_train, X_valid, y_valid):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=20, max_depth=4, max_features='sqrt', random_state=1)
    results = clf_model(rfc, X_train, y_train, X_valid)
    clf_model_evaluation(y_valid, results[0], results[1])
    model_rfc = results[2]
    word_importances = pd.DataFrame({'word': X_train.columns, 'importance': model_rfc.feature_importances_})
    print(word_importances.sort_values('importance', ascending=False).head(20))


def do_logistic_regression(y_train, X_train, X_valid, y_valid):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    results = clf_model(logreg, X_train, y_train, X_valid)
    clf_model_evaluation(y_valid, results[0], results[1])


# endregion
# endregion

# region Part C
def do_topic_modeling(data):
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    from gensim.parsing.preprocessing import preprocess_string

    texts = data.text.apply(preprocess_string).tolist()

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    from gensim.models.coherencemodel import CoherenceModel

    def calculate_coherence_score(documents, dictionary, model):
        coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
        return coherence_model.get_coherence()

    def get_coherence_values(start, stop):
        for num_topics in range(start, stop):
            print(f'\nCalculating coherence for {num_topics} topics')
            ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2)
            coherence = calculate_coherence_score(texts, dictionary, ldamodel)
            yield coherence

    min_topics, max_topics = 5, 15
    coherence_scores = list(get_coherence_values(min_topics, max_topics))

    import matplotlib.style as style

    style.use('fivethirtyeight')

    x = [int(i) for i in range(min_topics, max_topics)]

    ax = plt.figure(figsize=(10, 8))
    plt.xticks(x)
    plt.plot(x, coherence_scores)
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence Value')
    plt.title('Coherence Scores', fontsize=10)
    plt.show()


# endregion

# region Part D

def do_text_summarization(data):
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
        stop_words = stopwords.words('english')

        stop_words = stop_words + list(string.printable)
        words = [word for word in sentence if word not in stop_words]
        return [word for word in words if len(word) > 0]

    def tokenize_words(sentences):
        return [word_tokenize(sentence) for sentence in sentences]

    def fix_contractions(sentences):
        return [contractions.fix(sentence) for sentence in sentences]

    data['SentencesInArticle'] = data["text"].apply(sent_tokenize)
    data['WordsInSentences'] = data.SentencesInArticle.apply(fix_contractions).apply(lower).apply(tokenize_words).apply(
        remove_stopwords).apply(clean_sentences)

    data = data[['SentencesInArticle', 'WordsInSentences']]

    VECTOR_SIZE = 50
    EMPTY_VECTOR = np.zeros(VECTOR_SIZE)

    def sentence_vector(sentence):
        if (len(sentence) != 0):
            result = builtins.sum([glove_vectors.get(word, EMPTY_VECTOR) for word in sentence]) / len(sentence)
        else:
            result = builtins.sum([glove_vectors.get(word, EMPTY_VECTOR) for word in "empty sentence"]) / 2
        return result

    def sentences_to_vectors(sentences):
        return [sentence_vector(sentence) for sentence in sentences]

    data['SentenceVector'] = data.WordsInSentences.apply(sentences_to_vectors)

    def similarity_matrix(sentence_vectors):
        sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
        for i in range(len(sentence_vectors)):
            for j in range(len(sentence_vectors)):
                element_i = sentence_vectors[i].reshape(1, VECTOR_SIZE)
                element_j = sentence_vectors[j].reshape(1, VECTOR_SIZE)
                sim_mat[i][j] = cosine_similarity(element_i, element_j)[0, 0]
        return sim_mat

    data['SimMatrix'] = data['SentenceVector'].apply(similarity_matrix)

    def compute_graph(sim_matrix):
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        return scores

    data['Graph'] = data.SimMatrix.apply(compute_graph)

    def get_ranked_sentences(sentences, scores, n=3):
        top_scores = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        top_n_sentences = [sentence for score, sentence in top_scores[:n]]
        return " ".join(top_n_sentences)

    data['Summary'] = data.apply(lambda d: get_ranked_sentences(d.SentencesInArticle, d.Graph), axis=1)
    with open("summarized_texts/neg_summarized.txt", "a+") as f:
        f.write(data.loc[0].Summary)
    with open("summarized_texts/pos_summarized.txt", "a+") as f:
        f.write(data.loc[1].Summary)


# endregion

# region Part E
def train_sentiment_model(data):
    COLUMN_NAMES = ['text', 'sentiment']

    def clean(text):
        text = re.sub(r'[\W]+', ' ', text.lower())
        text = text.replace('hadn t', 'had not').replace('wasn t', 'was not').replace('didn t', 'did not')
        return text

    print(data.sentiment.value_counts())
    model_data = data.copy()
    model_data.Review = model_data.text.apply(clean)
    tfidf = TfidfVectorizer(strip_accents=None, preprocessor=None, lowercase=False)
    log_reg = LogisticRegression(random_state=0, solver='lbfgs')
    log_tfidf = Pipeline([('vect', tfidf), ('clf', log_reg)])
    X_train, X_test, y_train, y_test = train_test_split(model_data.text, model_data.sentiment, test_size=0.2,
                                                        random_state=42)

    log_tfidf.fit(X_train.values, y_train.values)
    test_accuracy = log_tfidf.score(X_test.values, y_test.values)
    print('The model has a test accuracy of {:.0%}'.format(test_accuracy))
    print(log_tfidf.predict(['I hated this place', 'I loved this place']))


# endregion

def main():
    initialize()
    neg_texts, pos_texts = read_files()
    data = create_data_frame_of_texts(neg_texts + pos_texts)
    clean_texts(data)

    def get_sentences():
        sentences_dict = {"text": [], "sentiment": []}
        for i in range(2000):
            sentences = sent_tokenize(data.loc[i]["text"])
            sentences_dict["text"].extend(sentences)
            for j in range(len(sentences)):
                sentences_dict["sentiment"].append(data.loc[i]["sentiment"])
        return pd.DataFrame(sentences_dict)

    tfidf_df = create_tfidf(data)
    feature_df = extract_features(data)
    print(feature_df.head())
    X_train, X_valid, y_train, y_valid = train_test_split(tfidf_df, data['sentiment'], test_size=0.2, random_state=42,
                                                          stratify=data['sentiment'])

    do_text_summarization(data.sample(5))


# We used this method to solve a problem occured in topic modeling that appears on Windows OS
if __name__ == '__main__':
    freeze_support()
    Process(target=main).start()
