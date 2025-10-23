import logging

import pandas as pd

# setting logger
logging.basicConfig(level=logging.INFO)


class ClassificationModel:
    def __init__(self):
        self.data = pd.read_csv('./data/cleaned_cuisines.csv')
        self.X = self.data.iloc[:, 2:]
        self.y = self.data[['cuisine']]
        self.model = None

    def train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        self.model = SVC(kernel='linear', C=10, probability=True, random_state=0)
        self.model.fit(X_train, y_train.values.ravel())

        y_pred = self.model.predict(X_test)
        logging.info(classification_report(y_test, y_pred))

    def transform_onnx(self):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, 380]))]
        options = {id(self.model): {'nocl': True, 'zipmap': False}}
        onx = convert_sklearn(self.model, initial_types=initial_type, options=options)
        with open("./static/output/model.onnx", "wb") as f:
            f.write(onx.SerializeToString())


if __name__ == '__main__':
    clf = ClassificationModel()
    clf.train()
    clf.transform_onnx()
