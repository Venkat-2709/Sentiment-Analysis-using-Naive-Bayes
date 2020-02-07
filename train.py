import json
import random

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB


def calcSentiment_train(trainFile):
    with open(trainFile, "r") as r:
        data = [json.loads(line) for line in r.readlines()]

    random.shuffle(data)

    re_list = []
    sent_list = []
    for i in data:
        re_1 = i.get("review")
        sent_1 = i.get("sentiment")
        re_list.append(re_1)
        sent_list.append(sent_1)

    sent_list = list(map(int, sent_list))

    tokenized_list = []
    review = []
    stop = set(stopwords.words('english'))
    for j in range(0, 802):
        ps = PorterStemmer()
        punctuation = RegexpTokenizer(r'\w+')
        tokenized_list.append(punctuation.tokenize(re_list[j]))
        stemmer = [ps.stem(word) for word in tokenized_list[j] if not word in stop]
        stemmed_sent = ' '.join(stemmer)
        review.append(stemmed_sent)

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(review).toarray()
    model = GaussianNB()
    model.fit(X, sent_list)

    return model, cv
