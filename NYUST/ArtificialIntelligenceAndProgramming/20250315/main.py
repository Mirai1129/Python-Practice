import pandas as pd
import numpy as np

# import datasets, tests and Support Vector Classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# if `load_breast_cancer(return_X_y=true)` -> only return X y -> tuple unpacking
# In this case, we use Bunch() -> attribute access
cancer = load_breast_cancer() # new cancer datasets

# Why X and y?
# X means the input features or independent variables. (The topic of the dataset)
# y means the target variable. (The answer of the dataset)
X = cancer.data
y = cancer.target

# train_test_split Returns: List containing train-test split of inputs.

# test_size: random seeds, different seed gives different result, so we set number for it.
# random_state: sklearn has a lot of random cause, we need random_state control the unsure issue.
# X_train, X_test: X training set and testing set
# y_train, y_test: y training set and testing set
# these var are using list unpacking.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

clf = SVC() # new Support Vector Classification
# We need to give SVC training set, so we use fit().
clf.fit(X_train, y_train)

# According to the trained model, we use X_test as topic then we get the predict answer.
y_predict = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
# Predict answer(y_predict) has to compare to correct answer(y_test).
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
# set index and columns.
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['predicted_cancer','predicted_healthy'])
print(classification_report(y_test, y_predict))
