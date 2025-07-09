# github_issue_scraper.py
"""
Fetches GitHub issues, assigns priority (High / Medium / Low),
trains a TF-IDF + logistic-regression model, and saves it to model/.

⚙️  Requirements
    pip install requests pandas scikit-learn python-dotenv  # (optional) imbalanced-learn

🔐  Optional: set a token for higher rate limits
    echo "GITHUB_TOKEN=ghp_yourPERSONALTOKEN" > .env
"""
import os, time, requests, pickle
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

###############################################################################
# 1. CONFIG
###############################################################################
REPOS            = ["microsoft/vscode", "tensorflow/tensorflow"]  # add more!
ISSUES_PER_REPO  = 400                                           # 400×2 = 800
OUT_DIR          = Path("model")
RAW_CSV          = Path("data/github_issues.csv")
###############################################################################

load_dotenv()                         # loads .env if present
TOKEN   = os.getenv("GITHUB_TOKEN")   # may be None
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

###############################################################################
# 2. PRIORITY MAPPING LOGIC
###############################################################################
LABEL_MAP = {
    2: ["p0", "blocker", "severity: critical", "severity/critical",
        "priority: 0", "priority: high"],
    1: ["p1", "severity: major", "severity/major", "priority: 1",
        "priority: medium"],
    0: ["p2", "p3", "severity: minor", "severity/minor",
        "priority: low", "trivial"]
}
KEYWORD_MAP = {
    2: ["crash", "blocker", "fatal", "urgent", "data loss"],
    1: ["error", "exception", "memory", "leak", "slow", "freeze", "hang"],
    0: []  # anything else
}

def priority_from_labels(labels):
    names = [l["name"].lower() for l in labels]
    for prio, patterns in LABEL_MAP.items():
        if any(p in n for n in names for p in patterns):
            return prio
    return None

def priority_from_text(text):
    text = text.lower()
    for prio, words in KEYWORD_MAP.items():
        if any(w in text for w in words):
            return prio
    return 0  # default Low

###############################################################################
# 3. GITHUB FETCH HELPERS
###############################################################################
def fetch_issues(repo, num):
    print(f"📡  Fetching {num} issues from {repo} …")
    collected, page = [], 1
    while len(collected) < num:
        url    = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "open", "per_page": 100, "page": page}
        res    = requests.get(url, headers=HEADERS, params=params, timeout=30)
        res.raise_for_status()
        batch  = [i for i in res.json() if "pull_request" not in i]
        if not batch:
            break
        collected.extend(batch)
        page += 1
        time.sleep(0.7)  # mild throttling
    return collected[:num]

###############################################################################
# 4. MAIN PIPELINE
###############################################################################
def main():
    # ---------------- collect + label ----------------
    all_rows = []
    for repo in REPOS:
        for issue in fetch_issues(repo, ISSUES_PER_REPO):
            prio = priority_from_labels(issue["labels"])
            if prio is None:
                text = f'{issue["title"]} {issue.get("body") or ""}'
                prio = priority_from_text(text)
            all_rows.append({
                "summary": issue["title"],
                "body": issue.get("body") or "",
                "priority": prio
            })



    df = pd.DataFrame(all_rows)
    RAW_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"  Saved raw issues → {RAW_CSV} │ class counts:\n{df.priority.value_counts()}")


    # ---------------- train model --------------------
    # ---------------- preprocess text -------------------
    
    df["full_text"] = (df["summary"] + " " + df["body"]).str.lower().str.replace(r"\s+", " ", regex=True)
    df = df[df["full_text"].str.len() > 15]


    vec = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words="english",
    max_df=0.85,
    min_df=2,
    max_features=15000
)

    X   = vec.fit_transform(df["full_text"])
    y   = df["priority"]

    model = LogisticRegression(max_iter=1500, class_weight="balanced")
    model.fit(X, y)

    OUT_DIR.mkdir(exist_ok=True)
    pickle.dump(model, (OUT_DIR / "bug_model.pkl").open("wb"))
    pickle.dump(vec,   (OUT_DIR / "tfidf_vectorizer.pkl").open("wb"))
    print("✅  Model + vectorizer saved in", OUT_DIR)


###############################################################################
# 5. ENTRY-POINT
###############################################################################
if __name__ == "__main__":
    main()
