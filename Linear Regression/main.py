import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

def train():
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(f"Accuracy: {acc*100}%")
    return linear

def save(model):
    # Writing to pickle file
    with open("studentmodel.pickle", "wb")  as f:
        pickle.dump(model, f)

def load():
    # Loading from pickle file as model
    pickle_in = open("studentmodel.pickle", "rb")
    linear = pickle.load(pickle_in)
    return linear

def test(model):
    print(f"Coefficients: \n {model.coef_}")
    print(f"Intercept: \n {model.intercept_}")

    predictions = model.predict(x_test)

    for x in range(len(predictions)):
        predictions[x] = int(predictions[x])
        print(f"Predicted: {predictions[x]} | Actual: {x_test[x]} {y_test[x]}")

def plot(x, y, data):
    style.use("ggplot")
    pyplot.scatter(data[x], data[y])
    pyplot.xlabel(x)
    pyplot.ylabel(y)
    pyplot.legend()

def main(mode):
    if mode == "train":
        model = train()
        save(model)
    elif mode == "test":
        model = load()
        test(model)
    elif mode == "plot":
        plot("absences", "G3", data)
        pyplot.show()

if __name__ == "__main__":
    main("plot")
