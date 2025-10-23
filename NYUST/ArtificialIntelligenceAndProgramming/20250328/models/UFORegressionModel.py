import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class UFORegressionModel:
    def __init__(self):
        self.model = None
        self.ufos = pd.read_csv('./data/ufos.csv')

    def preprocess_data(self):
        self.ufos = pd.DataFrame({
            'Seconds': self.ufos['duration (seconds)'],
            'Country': self.ufos['country'],
            'Latitude': self.ufos['latitude'],
            'Longitude': self.ufos['longitude']
        })
        self.ufos.dropna(inplace=True)
        self.ufos = self.ufos[(self.ufos['Seconds'] >= 1) & (self.ufos['Seconds'] <= 60)]
        self.ufos['Country'] = LabelEncoder().fit_transform(self.ufos['Country'])

    def train_model(self):
        X = self.ufos[['Seconds', 'Latitude', 'Longitude']]
        y = self.ufos['Country']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))
        print('Accuracy:', accuracy_score(y_test, predictions))

    def save_model(self, filename='ufo-model.pkl'):
        pickle.dump(self.model, open(f'./static/output/{filename}', 'wb'))

    def load_model(self, filename='ufo-model.pkl'):
        self.model = pickle.load(open(f'./static/output/{filename}', 'rb'))

    def predict(self, seconds, latitude, longitude):
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
        return self.model.ufo_predict([[seconds, latitude, longitude]])


if __name__ == '__main__':
    model = UFORegressionModel()
    model.preprocess_data()
    model.train_model()
    model.save_model()
    model.load_model()
