# Sentiment-Analysis-using-Naive-Bayes
## Predicting whether a review is positive or negative review using Naive Bayes Classification.

## Step 1: Importing the Libraries
```
NLTK 3.4.5 -----> pip install nltk (or) conda install nltk [For anaconda users]
Sklearn -----> pip install sklearn (or) conda install sklearn [For anaconda users]
```
## Step 2: Converting .csv to .json

Split the dataset into train and test set and then do the following to convert the csv to json file
```
import csv
import json

with open('filename.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with open('filename.json', 'w') as f:
    json.dump(rows, f)
```

## Step 3: Execute
```
Execute the main.py to see the results.
```
