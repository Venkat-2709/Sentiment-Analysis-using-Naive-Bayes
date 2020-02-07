import json

from test1 import calcSentiment_test
from train import calcSentiment_train

if __name__ == "__main__":
    trainFile = 'trainingFile.jsonlist'

    model, cv = calcSentiment_train(trainFile)
    print('Model Trained')

    result = []
    with open(trainFile, "r") as r:
        data = [json.loads(line) for line in r.readlines()]

    for line in data:
        review = line['review']
        result.append(calcSentiment_test(model, cv, review))

    print(result)
