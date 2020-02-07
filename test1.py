from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer


def calcSentiment_test(model, cv, review):
    corpus = []
    stop = set(stopwords.words('english'))
    ps = PorterStemmer()
    punctuation = RegexpTokenizer(r'\w+')
    review = punctuation.tokenize(review)
    stemmer = [ps.stem(word) for word in review if not word in stop]
    stemmed_sent = ' '.join(stemmer)
    corpus.append(stemmed_sent)
    X = cv.transform(corpus).toarray()

    sentiment = model.predict(X)

    if sentiment == 1:
        return True
    else:
        return False
