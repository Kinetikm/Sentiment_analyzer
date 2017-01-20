from sklearn.externals import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("class.pickle")
        self.vectorizer = joblib.load("vectorizer.pickle")
        self.classes_dict = {0: "негативный", 1: "позитивный", -1: "prediction error"}

    def predict_text(self, text):
        try:
            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            text = text.replace(',', '')
            text = text.replace('.', '')
            text = text.replace('!', '')
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)
        except:
            print("prediction error")
            return -1

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        return self.classes_dict[class_prediction]