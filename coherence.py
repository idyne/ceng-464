import pandas as pd
import warnings
from multiprocessing import Process, freeze_support


def main():
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_colwidth', 900)
    tweets = pd.read_csv('data/twitter-airline/Tweets.csv', usecols=['text'])
    tweets.head(10)
    import re

    HANDLE = '@\w+'
    LINK = 'https?://t\.co/\w+'
    SPECIAL_CHARS = '<|<|&|#'

    def clean(text):
        text = re.sub(HANDLE, ' ', text)
        text = re.sub(LINK, ' ', text)
        text = re.sub(SPECIAL_CHARS, ' ', text)
        return text

    tweets['text'] = tweets.text.apply(clean)
    tweets.head(10)
    from gensim.parsing.preprocessing import preprocess_string

    tweets = tweets.text.apply(preprocess_string).tolist()
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel

    dictionary = corpora.Dictionary(tweets)
    corpus = [dictionary.doc2bow(text) for text in tweets]
    NUM_TOPICS = 10
    ldamodel = LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.print_topics(num_words=6)

    from gensim.models.coherencemodel import CoherenceModel

    def calculate_coherence_score(documents, dictionary, model):
        if __name__ == "__main__":
            coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
            return coherence_model.get_coherence()
        else:
            print("error")

    def get_coherence_values(start, stop):
        for num_topics in range(start, stop):
            print(f'\nCalculating coherence for {num_topics} topics')
            ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2)
            coherence_model = CoherenceModel(model=ldamodel, texts=tweets, dictionary=dictionary, coherence='c_v')
            yield coherence_model.get_coherence()

    min_topics, max_topics = 10, 15
    coherence_scores = list(get_coherence_values(min_topics, max_topics))

    import matplotlib.pyplot as plt
    import matplotlib.style as style
    from matplotlib.ticker import MaxNLocator

    style.use('fivethirtyeight')

    x = [int(i) for i in range(min_topics, max_topics)]

    ax = plt.figure(figsize=(10, 8))
    plt.xticks(x)
    plt.plot(x, coherence_scores)
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence Value')
    plt.title('Coherence Scores', fontsize=10);
    plt.show()

    # plt.xaxis.set_major_locator(MaxNLocator(integer=True))


if __name__ == '__main__':
    freeze_support()
    Process(target=main).start()
