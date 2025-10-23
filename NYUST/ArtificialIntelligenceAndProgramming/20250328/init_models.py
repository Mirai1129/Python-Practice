from models import ClassificationModel, UFORegressionModel

def init_classification_model():
    clf = ClassificationModel()
    clf.train()
    clf.transform_onnx()

def init_ufo_regression_model():
    model = UFORegressionModel()
    model.preprocess_data()
    model.train_model()
    model.save_model()
    model.load_model()

def init_models_process():
    init_classification_model()
    init_ufo_regression_model()

if __name__ == '__main__':
    init_models_process()
