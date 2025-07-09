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
    if request.method == "POST":
        bug_text = request.form["bug_text"]
        vec = vectorizer.transform([bug_text])
        pred = model.predict(vec)[0]
        prediction = ["Low", "Medium", "High"][pred]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
