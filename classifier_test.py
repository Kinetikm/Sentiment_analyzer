
# coding: utf-8

from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

pred = clf.get_prediction_message("Абсолютно ужасный телефон не советую никому его покупок, весь сплошной недостаток")

print(pred)