import requests, os, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

GITHUB_REPO = "microsoft/vscode"  # or "tensorflow/tensorflow"
NUM_ISSUES = 200
TOKEN = os.getenv("GITHUB_TOKEN")  # optional

headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

def fetch_issues():
    issues = []
    page = 1
    while len(issues) < NUM_ISSUES:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
        params = {"state": "open", "per_page": 100, "page": page}
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            raise Exception(f"Failed to fetch: {r.status_code} {r.text}")
        page_data = r.json()
        if not page_data:
            break
        issues += [i for i in page_data if "pull_request" not in i]
        page += 1
        time.sleep(1)  # rate limiting
    return issues[:NUM_ISSUES]

def map_priority(text):
    text = text.lower()
    if any(w in text for w in ["crash", "block", "fatal", "urgent"]):
        return 2  # High
    elif any(w in text for w in ["slow", "error", "fail", "memory", "exception"]):
        return 1  # Medium
    else:
        return 0  # Low

def main():
    print("📡 Fetching issues...")
    issues = fetch_issues()
    print(f"Fetched {len(issues)} issues.")

    df = pd.DataFrame({
        "summary": [i["title"] for i in issues],
        "priority": [map_priority(i["title"] + " " + (i.get("body") or "")) for i in issues]
    })

    # Save raw data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/github_issues.csv", index=False)

    # Train model
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["summary"])
    y = df["priority"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    pickle.dump(model, open("model/bug_model.pkl", "wb"))
    pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

    print(" Model trained and saved from live GitHub issues.")

if __name__ == "__main__":
    main()
