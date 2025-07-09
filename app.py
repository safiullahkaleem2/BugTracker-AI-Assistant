from flask import Flask, request, render_template
import pickle
import os

# Load model and vectorizer
model_path = os.path.join("model", "bug_model.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    if request.method == "POST":
        bug_text = request.form["bug_text"].strip()
        vec_input = vectorizer.transform([bug_text])
        if vec_input.nnz == 0:
            prediction = "Unrecognized or gibberish input"
        else:
            pred = model.predict(vec_input)[0]
            proba = model.predict_proba(vec_input)[0]
            prediction = ["Low", "Medium", "High"][pred]
            print(f"Proba: {proba}")



    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
