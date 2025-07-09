#  BugTracker AI Assistant

A machine learning-powered tool that classifies GitHub bug reports into **Low**, **Medium**, or **High** priority using real-world issue data. Designed with a Flask web dashboard for non-technical users.

---

##  Features

- Fetches live bug reports from GitHub repositories
- Heuristically and label-based priority classification
- Trains TF-IDF + Logistic Regression model
- Handles invalid/gibberish input gracefully
- Flask interface for submitting and triaging issues
