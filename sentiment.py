# %% [markdown]
# # Sentiment Analysis Yelp

# %%
import pandas as pd

pd.set_option('display.max_colwidth', 200)

# %%
DATA_DIR = 'data/sentiment_labelled_sentences/'

IMDB_DATA_FILE = DATA_DIR + 'imdb_labelled.txt'
YELP_DATA_FILE = DATA_DIR + 'yelp_labelled.txt'
AMAZON_DATA_FILE = DATA_DIR + 'amazon_cells_labelled.txt'

COLUMN_NAMES = ['Review', 'Sentiment']

yelp_reviews = pd.read_table(YELP_DATA_FILE, names=COLUMN_NAMES)
amazon_reviews = pd.read_table(AMAZON_DATA_FILE, names=COLUMN_NAMES)
imdb_reviews = pd.read_table(YELP_DATA_FILE, names=COLUMN_NAMES)

review_data = pd.concat([amazon_reviews, imdb_reviews, yelp_reviews])

review_data.sample(10)

review_data.Sentiment.value_counts()

import re


def clean(text):
    text = re.sub(r'[\W]+', ' ', text.lower())
    text = text.replace('hadn t', 'had not').replace('wasn t', 'was not').replace('didn t', 'did not')
    return text


review_model_data = review_data.copy()
review_model_data.Review = review_model_data.Review.apply(clean)

review_model_data.sample(10)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(strip_accents=None, preprocessor=None, lowercase=False)
log_reg = LogisticRegression(random_state=0, solver='lbfgs')
log_tfidf = Pipeline([('vect', tfidf), ('clf', log_reg)])

X_train, X_test, y_train, y_test = train_test_split(review_model_data.Review, review_model_data.Sentiment,
                                                    test_size=0.3, random_state=42)

log_tfidf.fit(X_train.values, y_train.values)

test_accuracy = log_tfidf.score(X_test.values, y_test.values)
'The model has a test accuracy of {:.0%}'.format(test_accuracy)
print(log_tfidf.predict(['I loved this place', 'I hated this place']))
