import pickle

import numpy as np
from flask import Flask, request, render_template

from init_models import init_models_process

app = Flask(__name__)

init_models_process()
model = pickle.load(open("./static/output/ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/cuisines")
def cuisines():
    return render_template("cuisines.html")


@app.route("/ufo")
def ufo():
    return render_template("ufo.html")


@app.route("/ufo/predict", methods=["POST"])
def ufo_predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "ufo.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
